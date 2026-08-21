"""The margin trainer: fit the embedder to the teacher's scores directly.

One of two training loops in this package. This one takes the cross-encoder's score as a
regression target -- the teacher said 0.83, so the student should say 0.83 -- and is what
`lara fit-check` and `lara lr-sweep` drive.

The other is lara/finetune/mnrl.py, which learns from ranking rather than from magnitude.
They share the Recipe, the data and the evaluation, and nothing else; they were in one
951-line module called `kfold`, which described a twelve-line function in it.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from lara import device as dev
from lara.index.prefixes import QUERY_PROMPT as QUERY_PREFIX

from lara.finetune.triples import Triple, doc_input, inner_split  # noqa: F401


def _encode(model, texts: list[str], device: str, grad: bool = True) -> torch.Tensor:
    features = {k: v.to(device) for k, v in model.tokenize(texts).items()}
    if grad:
        return model(features)["sentence_embedding"]
    with torch.no_grad():
        return model(features)["sentence_embedding"]



def evaluate(model, triples: list[Triple], device: str, batch: int = 48,
             whitener=None) -> dict:
    """Pairwise metrics against the teacher.

    ``whitener`` applies the corpus whitening transform to every embedding before the
    similarities are taken, so a run can be scored in the space retrieval would actually
    use. It is applied to queries and documents with the same matrix: they are compared to
    each other, and transforming only one side would put them in different spaces.
    """
    if not triples:
        return {"n": 0}
    model.eval()
    student, teacher = [], []
    for i in range(0, len(triples), batch):
        part = triples[i:i + batch]
        q = _encode(model, [QUERY_PREFIX + t.query for t in part], device, grad=False)
        p = _encode(model, [doc_input(t.pos_text) for t in part], device, grad=False)
        n = _encode(model, [doc_input(t.neg_text) for t in part], device, grad=False)
        if whitener is not None:
            import numpy as _np
            qa = whitener(q.float().cpu().numpy())
            pa = whitener(p.float().cpu().numpy())
            na = whitener(n.float().cpu().numpy())
            m_np = (qa * pa).sum(-1) - (qa * na).sum(-1)
            student.extend([float(x) for x in _np.asarray(m_np)])
            teacher.extend(t.margin for t in part)
            continue
        qn = F.normalize(q.float(), dim=-1)
        m = (qn * F.normalize(p.float(), dim=-1)).sum(-1) \
          - (qn * F.normalize(n.float(), dim=-1)).sum(-1)
        student.extend(m.cpu().tolist())
        teacher.extend(t.margin for t in part)
    model.train()

    s = np.asarray(student)
    t = np.asarray(teacher)
    rank = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    rs, rt = rank(s), rank(t)
    spearman = float(np.corrcoef(rs, rt)[0, 1]) if len(s) > 2 else 0.0

    # Within-query rank correlation, averaged over queries.
    #
    # `spearman` above pools every triple from every query into one ranking, which asks
    # whether query A's margin is bigger than query B's — a comparison retrieval never
    # makes. Ordering candidates *for one question* is the thing retrieval does, and it is
    # a strictly harder problem: measured on the base model, pooled 0.695 against
    # within-query 0.526 +/- 0.295, with 7% of queries scoring negative. Reporting only the
    # pooled figure overstates the model and hides that spread.
    by_q: dict[str, list[int]] = {}
    for i, tr in enumerate(triples):
        by_q.setdefault(tr.query_hash, []).append(i)
    per_q = []
    for idx in by_q.values():
        if len(idx) < 4:
            continue
        a, b = s[idx], t[idx]
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            continue
        per_q.append(float(np.corrcoef(rank(a), rank(b))[0, 1]))

    return {
        "n": len(s),
        "pair_acc": float((s > 0).mean()),
        "margin_mae": float(np.abs(s - t).mean()),
        "mean_student_margin": float(s.mean()),
        "mean_teacher_margin": float(t.mean()),
        "spearman": round(spearman, 4),
        "within_q": round(float(np.mean(per_q)), 4) if per_q else None,
        "within_q_std": round(float(np.std(per_q)), 4) if per_q else None,
        "n_queries_scored": len(per_q),
    }



@dataclass
class Recipe:
    """Fixed after the first run destroyed the encoder.

    The original used Muon at 2e-3 with batch 16. Muon normalises its update per matrix, so
    the effective step ignores how large the pretrained weights already are — a pretraining
    LR applied to fine-tuning. Both evals collapsed ~85% while the loss barely moved, the
    signature of destroying pretrained structure rather than learning the task.

    ``epochs`` is now a *cap*, not a target: with ``patience`` set, training stops when the
    inner-validation loss stops improving. The first 5-fold run trained a fixed 3 epochs and
    degraded rank correlation in all five folds while the loss it was optimising kept
    falling — the signature of running past the useful point.
    """
    lr_muon: float = 5e-5          # was 2e-3 — 40x lower
    lr_adam: float = 1e-5          # was 2e-5
    batch_size: int = 128          # was 16, then 64 — steadier gradient still
    # Sequences per forward pass. `batch_size` is the OPTIMISER batch and is reached by
    # accumulating gradients over micro-batches, because one step encodes three sequences
    # per triple — query, positive, negative — so batch 512 is 1,536 sequences and does
    # not fit alongside its own backward pass on a 96 GB card. Accumulation makes the
    # optimiser batch a free parameter instead of a memory limit; 0 disables it.
    micro_batch: int = 64
    epochs: int = 4                # upper bound; early stopping decides the real number
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    max_seq_length: int = 320
    grad_clip: float = 1.0
    margin_scale: float = 10.0
    # InfoNCE scale for MultipleNegativesRankingLoss. 0.05 == logits x20, matching
    # `embedding.temperature` used for the citation objective in train.py.
    temperature: float = 0.05
    # Sharpness-Aware Minimisation. 0 disables it. SAM takes the gradient at the worst
    # point in an epsilon-ball around the current weights rather than at the weights
    # themselves, which biases training toward flat minima and is worth most exactly where
    # we are: 20 k examples from 788 queries against a 300 M encoder, early-stopping at
    # step ~95 of 148 because validation turns over. Overfitting is the observed failure,
    # and flat minima are the thing that addresses it.
    #
    # It costs a second full gradient per step, and each of ours is already a two-pass
    # GradCache cycle, so a SAM step is four forwards rather than two. 0.05 is the value
    # the paper uses for fine-tuning; ASAM's adaptive radius is the obvious next knob if
    # this helps.
    sam_rho: float = 0.0
    # Exponential moving average of the weights, evaluated instead of the raw ones. Nearly
    # free — one extra copy of the parameters and a lerp per step — and it captures much of
    # what averaging-based regularisation offers without a second gradient.
    ema_decay: float = 0.0
    #: Cap on triples used for the early-stopping signal.
    #:
    #: The inner split is a FRACTION of the training set, so it grew with every harvest —
    #: 3,649 triples at 24 k, 24,116 at 161 k. At 161 k one validation encoded 71,424
    #: sequences against 1,536 for a training step: 46.5x a step, every 10 steps, so 82 %
    #: of all compute went to deciding when to stop rather than to training.
    #:
    #: Early stopping needs a stable, representative estimate, not an exhaustive one. A
    #: fixed subsample gives the same decision at a small fraction of the cost, and the
    #: subsample is deterministic so the loss stays comparable between evaluations.
    val_max_triples: int = 2048
    compile_mode: str | None = None   # off by default: folds are short, compile is not free
    seed: int = 0

    # ── early stopping ────────────────────────────────────────────────────────────
    # Evaluated against an INNER split carved out of the training data, never against the
    # held-out fold: selecting a checkpoint on the fold you then report would leak, and the
    # reported numbers would be optimistic by an unknown amount.
    #
    # **Keep `epochs` realistic rather than a large cap.** The cosine LR schedule is sized
    # to `epochs`, so a big cap means stopping mid-schedule at a near-peak learning rate.
    # Measured: epochs=3 anneals to 1.3e-6 by its final step and scores pair_acc 0.8526;
    # epochs=12 stops around step 65 of 216 with LR still at 4.4e-5 and scores 0.8472,
    # despite training *longer*. Cosine annealing and early stopping are two mechanisms for
    # the same job, and an unannealed checkpoint loses more than the extra steps gain.
    # Early stopping is best used here as a safety net, not as the primary control.
    patience: int = 3              # evaluations without improvement before stopping
    eval_every: int = 5            # steps between validation passes
    inner_val_frac: float = 0.15   # of the training set, split by query
    min_delta: float = 1e-3        # improvement smaller than this does not count



def validation_loss(model, triples: list[Triple], device: str, rec: Recipe,
                    batch: int = 64) -> float:
    """Mean MarginMSE over a held-out slice — the same quantity training minimises.

    Deliberately the *training objective* rather than a ranking metric: early stopping
    should answer "has this stopped learning", and mixing in a differently-scaled metric
    makes the patience threshold meaningless.
    """
    from lara.finetune.train import margin_mse

    if not triples:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(triples), batch):
            part = triples[i:i + batch]
            q = _encode(model, [QUERY_PREFIX + t.query for t in part], device, grad=False)
            p = _encode(model, [doc_input(t.pos_text) for t in part], device, grad=False)
            ng = _encode(model, [doc_input(t.neg_text) for t in part], device, grad=False)
            tm = torch.tensor([t.margin for t in part], device=device)
            loss = margin_mse(q.float(), p.float(), ng.float(), tm, rec.margin_scale)
            total += float(loss) * len(part)
            n += len(part)
    model.train()
    return total / max(1, n)



def train_on(triples: list[Triple], model_name: str, device: str, rec: Recipe,
             progress=None, val_triples: list[Triple] | None = None):
    """Train one model on one set of triples with MarginMSE.

    If ``val_triples`` is given, validation loss is checked every ``rec.eval_every`` steps,
    the best-scoring weights are kept, and training stops after ``rec.patience``
    evaluations without improvement. The best checkpoint is restored before returning — not
    the last one, which is the whole point of measuring.
    """
    from lara.finetune.optim import load_muon

    SingleDeviceMuonWithAuxAdam = load_muon()
    from sentence_transformers import SentenceTransformer

    from lara.finetune.train import margin_mse, split_param_groups

    torch.manual_seed(rec.seed)
    device = dev.resolve(device)
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = rec.max_seq_length
    model.train()
    inner = model[0].auto_model
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    inner.config.use_cache = False
    if rec.compile_mode:
        model[0].auto_model = torch.compile(inner, mode=rec.compile_mode, dynamic=True)

    class _Cfg:
        lr_muon, lr_adam, weight_decay = rec.lr_muon, rec.lr_adam, rec.weight_decay
    opt = SingleDeviceMuonWithAuxAdam(split_param_groups(model, _Cfg()))

    steps = max(1, (len(triples) // rec.batch_size) * rec.epochs)
    warmup = max(1, int(steps * rec.warmup_frac))
    base = [g["lr"] for g in opt.param_groups]
    rng = random.Random(rec.seed)
    step = 0
    started = time.time()

    best_loss, best_state, since_best, stopped = float("inf"), None, 0, False
    val_triples = val_triples or []

    def snapshot():
        # Kept on CPU: a second fp32 copy of a 300M-param encoder is ~1.2 GB, which is
        # cheap in host RAM and would otherwise compete with activations on the device.
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    for _ in range(rec.epochs):
        if stopped:
            break
        rng.shuffle(triples)
        for i in range(0, len(triples) - rec.batch_size + 1, rec.batch_size):
            part = triples[i:i + rec.batch_size]
            scale = (step / warmup) if step < warmup else 0.5 * (
                1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale

            # Accumulate to the full optimiser batch. Each micro-batch's mean loss is
            # weighted by its share of the batch, so the accumulated gradient equals the
            # gradient of the mean over the whole batch — not the mean of per-micro-batch
            # gradients, which differ whenever the last micro-batch is short.
            micro = rec.micro_batch or len(part)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for j in range(0, len(part), micro):
                sub_part = part[j:j + micro]
                with dev.autocast(device):
                    q = _encode(model, [QUERY_PREFIX + t.query for t in sub_part], device)
                    p = _encode(model, [doc_input(t.pos_text) for t in sub_part], device)
                    n = _encode(model, [doc_input(t.neg_text) for t in sub_part], device)
                    tm = torch.tensor([t.margin for t in sub_part], device=device)
                    loss = margin_mse(q.float(), p.float(), n.float(), tm, rec.margin_scale)
                (loss * (len(sub_part) / len(part))).backward()
                total += float(loss.detach()) * (len(sub_part) / len(part))
            torch.nn.utils.clip_grad_norm_(model.parameters(), rec.grad_clip)
            opt.step()
            step += 1
            loss = torch.tensor(total)

            vl = None
            if val_triples and step % rec.eval_every == 0:
                vl = validation_loss(model, val_triples, device, rec)
                if vl < best_loss - rec.min_delta:
                    best_loss, best_state, since_best = vl, snapshot(), 0
                else:
                    since_best += 1

            if progress is not None and step % 5 == 0:
                progress.send({"step": step, "steps": steps, "loss": float(loss.detach()),
                               "lr": opt.param_groups[0]["lr"], "val_loss": vl,
                               "best_val": best_loss if best_state is not None else None,
                               "elapsed": time.time() - started})

            if val_triples and since_best >= rec.patience:
                if progress is not None:
                    progress.send({"step": step, "steps": steps,
                                   "loss": float(loss.detach()),
                                   "lr": opt.param_groups[0]["lr"], "val_loss": vl,
                                   "best_val": best_loss, "early_stop": True,
                                   "elapsed": time.time() - started})
                stopped = True
                break

    # Restore the best checkpoint, not the last — otherwise the measurement was pointless.
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    model.stopped_early = stopped
    model.best_val_loss = best_loss if best_state is not None else None
    model.steps_trained = step
    return model



def sweep(triples: list[Triple], model_name: str, device: str, rec: Recipe,
          lrs: list[float], *, val_frac: float = 0.25, seed: int = 0,
          progress=None) -> list[dict]:
    """Learning-rate sweep at a fixed recipe, scored on held-out queries.

    Each LR trains from the same pretrained checkpoint on the same split, so the only
    difference between runs is the learning rate. The split is **by query**, matching the
    outer folds: sharing a query across train and validation would let a run score well by
    memorising a paraphrase of the thing it is being tested on.

    ``lr_adam`` is scaled with ``lr_muon`` rather than held fixed. The two exist in a ratio
    the recipe already chose (5:1); pinning Adam while sweeping Muon would silently change
    that ratio across the sweep and confound "which learning rate" with "which balance
    between the two optimisers".

    Returns one record per LR, in the order given, each carrying the before/after metrics
    on the held-out slice and what early stopping did.
    """
    train, val = inner_split(triples, val_frac, seed)
    if not val:
        raise ValueError("val split is empty — too few distinct queries")

    from lara.index.embed import load_model

    base = load_model(model_name, device=dev.resolve(device), max_seq_length=rec.max_seq_length)
    before = evaluate(base, val, dev.resolve(device))
    del base
    dev.empty_cache(device)

    ratio = rec.lr_adam / rec.lr_muon if rec.lr_muon else 0.2
    out: list[dict] = []
    for lr in lrs:
        r = replace(rec, lr_muon=lr, lr_adam=lr * ratio, seed=seed)
        # The inner split is carved from `train` only, so the sweep's validation slice
        # never influences early stopping *and* never influences the reported number.
        inner_train, inner_val = inner_split(train, rec.inner_val_frac, seed + 1)
        model = train_on(list(inner_train), model_name, device, r,
                         progress=progress, val_triples=inner_val)
        after = evaluate(model, val, dev.resolve(device))
        out.append({
            "lr_muon": lr,
            "lr_adam": lr * ratio,
            "steps": getattr(model, "steps_trained", None),
            "early_stopped": bool(getattr(model, "stopped_early", False)),
            "best_val_loss": getattr(model, "best_val_loss", None),
            "before": before,
            "after": after,
            "d_pair_acc": after["pair_acc"] - before["pair_acc"],
            "d_spearman": (after["spearman"] - before["spearman"])
                          if (after["spearman"] is not None and before["spearman"] is not None)
                          else None,
            "d_margin_mae": after["margin_mae"] - before["margin_mae"],
        })
        del model
        dev.empty_cache(device)
    return out



def format_sweep(rows: list[dict]) -> str:
    """One line per learning rate. Deltas, because the absolute numbers hide small moves."""
    if not rows:
        return "(no sweep rows)"
    b = rows[0]["before"]
    head = (f"baseline on held-out queries: pair_acc {b['pair_acc']:.4f}  "
            f"spearman {b['spearman']:.4f}  margin_mae {b['margin_mae']:.4f}\n\n"
            f"{'lr_muon':>9} {'steps':>6} {'stop':>5} "
            f"{'pair_acc':>18} {'spearman':>18} {'margin_mae':>18}\n")
    lines = []
    for r in rows:
        a = r["after"]
        sp = f"{a['spearman']:.4f} ({r['d_spearman']:+.4f})" if r["d_spearman"] is not None else "n/a"
        lines.append(
            f"{r['lr_muon']:>9.1e} {str(r['steps']):>6} {'yes' if r['early_stopped'] else 'no':>5} "
            f"{a['pair_acc']:.4f} ({r['d_pair_acc']:+.4f}) {sp:>18} "
            f"{a['margin_mae']:.4f} ({r['d_margin_mae']:+.4f})"
        )
    return head + "\n".join(lines)

