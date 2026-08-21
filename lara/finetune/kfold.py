"""K-fold cross-validation over harvested judgements.

Answers a narrower question than the retrieval eval does, and a more urgent one: **can the
model fit this data at all?** After a fine-tune that destroyed the encoder, the first thing
to establish is whether the training loop can learn anything, before asking whether what it
learns generalises.

So there are two modes:

``overfit``  train and evaluate on the SAME few hundred triples. A working setup drives
             pairwise accuracy toward 1.0 here. If it cannot overfit a tiny set, the loss,
             the optimiser or the learning rate is wrong, and no amount of extra data will
             help — this is the check that would have caught the 2e-3 Muon LR in minutes
             instead of after a 21-minute run and a wrecked model.
``kfold``    split by QUERY, train on k-1 folds, evaluate on the held-out one. Splitting by
             triple would leak: the same query contributes many triples sharing its
             phrasing, so a model could score well on the test half by memorising the
             train half of the same question.

Metrics are pairwise, not ranking-based, because that is what the training signal is:

``pair_acc``   fraction of (positive, negative) pairs the student orders correctly
``margin_mae`` mean absolute error against the teacher's gap — how well the *degree* of
               relevance was reproduced, not merely the sign
``spearman``   rank correlation with the teacher over all pairs
"""

from __future__ import annotations

import math
import random
import sqlite3
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F

from lara import device as dev

# Imported, not retyped: these must match what the corpus was embedded with, and four
# hand-written copies had already drifted apart. See lara.index.embed.
from lara.index.embed import DOC_PREFIX_UNTITLED as DOC_PREFIX  # noqa: E402
from lara.index.embed import QUERY_PROMPT as QUERY_PREFIX  # noqa: E402


@dataclass
class Triple:
    query: str
    query_hash: str
    pos_text: str
    neg_text: str
    margin: float          # teacher's positive-minus-negative score


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


def canon_query(q: str) -> str:
    """A grouping key that survives trailing punctuation and casing.

    `judgements.query_hash` is a SHA of the whitespace-normalised, lowercased question, and
    the unique index on (query_hash, chunk_id, teacher) stops the same judgement being
    stored twice. What it cannot catch is the same question stored under two hashes:
    measured, "What optimizer does scaled dot-product attention use?" and the same string
    without the "?" are distinct rows at cosine 0.998.

    That matters more for MNRL than for storage. Two groups that are really one question
    can land in the same batch, and then one group's positive is a false negative for the
    other — exactly the failure `query_disjoint_batches` exists to prevent. Re-grouping
    here fixes old and new rows alike and needs no migration, whereas changing `qhash`
    itself would split every existing question from its future judgements.
    """
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in q).lower().split())


def _render(item: dict, contextual: bool) -> str:
    """The document string, in the format the reader will actually be searched in.

    ``contextual=True`` reproduces `lara.index.embed.document_text`, which is what the
    29.5 M corpus vectors were built with. False keeps the bare-chunk form the earlier runs
    used, so old results stay reproducible.

    The returned string is already a complete EmbeddingGemma document prompt, so callers
    must not prepend DOC_PREFIX again — `doc_input` below is what enforces that.
    """
    if not contextual:
        return item["text"]
    from lara.index.embed import document_text
    return document_text(item.get("paper_title"), item.get("section_title"), item["text"])


def doc_input(text: str) -> str:
    """Prepend the document prompt unless the text already carries one."""
    return text if text.startswith("title:") else DOC_PREFIX + text


def make_triples(conn: sqlite3.Connection, max_per_query: int = 8,
                 min_margin: float = 0.05, limit: int = 0,
                 hard_frac: float = 0.5, contextual: bool = False) -> list[Triple]:
    """Pair every positive with a negative from the same query.

    ``min_margin`` drops pairs the teacher itself could barely separate: near-zero gaps
    are where the reranker is guessing, and asking the student to reproduce a coin flip
    adds variance without signal.

    It was 0.2, which turned out to be too generous. The overfit check opened at pair_acc
    0.963 — the untrained model already ordered almost every pair correctly, so that
    metric measured nothing and only ``margin_mae`` carried information. A lower floor
    admits genuinely hard pairs.

    ``hard_frac`` reserves half of each query's budget for the negatives the teacher
    scored HIGHEST — the ones retrieval surfaced and the reranker still rejected. Those
    are the discriminations worth learning; a positive against a random chunk is a
    distinction the model already makes.
    """
    from lara.finetune.judgements import training_pairs

    out: list[Triple] = []
    for group in training_pairs(conn):
        pos = sorted(group["positives"], key=lambda x: -(x["score"] or 0))
        neg_all = sorted(group["negatives"], key=lambda x: -(x["score"] or 0))
        if not pos or not neg_all:
            continue
        # Hardest negatives first, then the easiest, so each query contributes both.
        n_hard = max(1, int(len(neg_all) * hard_frac))
        neg = neg_all[:n_hard] + neg_all[n_hard:][::-1]
        from lara.finetune.judgements import qhash
        qh = qhash(group["query"])
        made = 0
        for p in pos:
            for n in neg:
                margin = (p["score"] or 0) - (n["score"] or 0)
                if margin < min_margin:
                    continue
                out.append(Triple(
                    group["query"], canon_query(group["query"]),
                    _render(p, contextual), _render(n, contextual), float(margin)))
                made += 1
                if made >= max_per_query:
                    break
            if made >= max_per_query:
                break
        if limit and len(out) >= limit:
            break

    # Identical (query, positive, negative) can be produced more than once when a chunk is
    # judged under two teachers or a question was harvested twice. 350 of 84,874 measured —
    # small, but a duplicate triple is pure redundant compute and, under an in-batch loss,
    # a guaranteed false negative if both copies land in one batch.
    seen: set[tuple[str, str, str]] = set()
    unique: list[Triple] = []
    for t in out:
        key = (t.query_hash, t.pos_text, t.neg_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def split_by_query(triples: list[Triple], k: int, seed: int = 0) -> list[list[int]]:
    """Assign fold indices grouped by query, never by triple."""
    rng = random.Random(seed)
    queries = sorted({t.query_hash for t in triples})
    rng.shuffle(queries)
    fold_of = {q: i % k for i, q in enumerate(queries)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, t in enumerate(triples):
        folds[fold_of[t.query_hash]].append(i)
    return folds


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


def inner_split(triples: list[Triple], frac: float, seed: int = 0
                ) -> tuple[list[Triple], list[Triple]]:
    """Carve a validation slice off the training set, **split by query**.

    Same anti-leak rule as the outer folds: one query contributes many triples sharing its
    phrasing, so splitting by triple would let the model see a paraphrase of every
    validation item during training and make the stopping signal useless.
    """
    rng = random.Random(seed)
    queries = sorted({t.query_hash for t in triples})
    rng.shuffle(queries)
    n_val = max(1, int(round(len(queries) * frac)))
    val_q = set(queries[:n_val])
    train = [t for t in triples if t.query_hash not in val_q]
    val = [t for t in triples if t.query_hash in val_q]
    return (train, val) if train and val else (triples, [])


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
    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:      # noqa: F401 - re-raised with instructions
        raise ImportError(
            "the Muon optimiser is not installed. It is not on PyPI under that name -- "
            "`pip install muon` fetches an unrelated omics package that will import but "
            "not provide SingleDeviceMuonWithAuxAdam. Install from source:\n"
            "    pip install git+https://github.com/KellerJordan/Muon"
        ) from exc
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


# ── MultipleNegativesRankingLoss, with GradCache ─────────────────────────────────────

def mnrl(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
         temperature: float = 0.05) -> torch.Tensor:
    """InfoNCE over in-batch negatives plus the mined hard negative.

    This is `MultipleNegativesRankingLoss`, the loss Google's own EmbeddingGemma guide
    recommends, and it optimises *order* rather than the magnitude of a score gap — which
    is what every metric the MarginMSE runs failed on was measuring.

    Candidates for row i are all B positives and all B mined negatives, with the correct
    answer at index i. So a batch of 512 supplies 1,023 negatives per query: 511 other
    queries' positives, which are easy, plus 512 reranker-rejected passages, which are the
    hard ones. Batch size stops being a smoothing knob and becomes the supply of negatives.
    """
    q = F.normalize(q, dim=-1)
    cand = F.normalize(torch.cat([pos, neg], dim=0), dim=-1)   # (2B, D)
    logits = (q @ cand.T) / temperature                        # (B, 2B)
    target = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, target)


def query_disjoint_batches(triples: list[Triple], batch_size: int,
                           rng: random.Random) -> list[list[Triple]]:
    """Batches in which no query appears twice.

    In-batch negatives assume every other row is genuinely irrelevant. That is false here:
    one query contributes up to 32 triples, so two triples of the same query put a *real*
    positive for that query into its own negative set. Measured on a random batch of 512,
    137 rows — 27 % — collided that way, and InfoNCE spends those rows teaching the model
    to push away passages it should be retrieving.

    MarginMSE never cared, because each triple carried its own negative and the batch was
    only a gradient-smoothing device. Switching to a batch-is-the-negatives loss makes the
    sampler part of the objective.

    Round-robin over per-query queues, so a query with many triples contributes to many
    batches but never twice to one.
    """
    by_q: dict[str, list[Triple]] = {}
    for t in triples:
        by_q.setdefault(t.query_hash, []).append(t)
    queues = list(by_q.values())
    for q in queues:
        rng.shuffle(q)
    batches: list[list[Triple]] = []
    while True:
        live = [q for q in queues if q]
        if len(live) < batch_size:
            break                      # a short final batch would have fewer negatives
        rng.shuffle(live)
        batches.append([q.pop() for q in live[:batch_size]])
    return batches


def length_sorted_chunks(part: list[Triple], micro: int) -> list[list[Triple]]:
    """Split a batch into micro-batches of similar length.

    Tokenisation pads each micro-batch to its longest member, and document lengths here
    run mean 260 tokens against a p95 of 512. Measured on random micro-batches of 64, that
    padding costs a factor of **1.98** — about half of every step is spent on positions
    that are not there.

    Sorting the batch by length before splitting puts similar-length documents together,
    so each micro-batch pads to nearly its own mean instead of the batch-wide maximum.

    **This does not change the loss.** The permutation is applied identically to queries,
    positives and negatives, so row i still pairs with row i and the InfoNCE target is the
    same diagonal over the same set. Only the grouping into forward passes changes, and
    the loss is computed after concatenation.

    Character length is used rather than token count: it correlates closely enough to sort
    by, and tokenising the batch twice to save tokenising it once would be self-defeating.
    """
    order = sorted(part, key=lambda t: max(len(t.pos_text), len(t.neg_text)))
    return [order[i:i + micro] for i in range(0, len(order), micro)]


def _encode_ids(model, feats: dict) -> torch.Tensor:
    return model(feats)["sentence_embedding"]


def train_on_mnrl(triples: list[Triple], model_name: str, device: str, rec: "Recipe",
                  progress=None, val_triples: list[Triple] | None = None):
    """Train with MultipleNegativesRankingLoss at a batch size memory cannot hold.

    **Gradient accumulation is wrong for this loss.** Under MarginMSE each triple carries
    its own negative, so accumulating eight micro-batches of 64 reproduces the 512 gradient
    exactly. Under InfoNCE the batch *is* the negative set, and the same accumulation
    computes eight independent 64-way problems — a weaker objective wearing a large batch's
    clothing.

    So this uses GradCache instead. Encode every micro-batch under ``no_grad`` and cache
    the embeddings; compute the real 512-way loss on the cache and get its gradient with
    respect to those embeddings; then re-encode each micro-batch with graph and seed
    backward with the cached gradient. The result is the exact full-batch gradient at the
    memory cost of one micro-batch, paid for with a second forward pass.
    """
    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:      # noqa: F401 - re-raised with instructions
        raise ImportError(
            "the Muon optimiser is not installed. It is not on PyPI under that name -- "
            "`pip install muon` fetches an unrelated omics package that will import but "
            "not provide SingleDeviceMuonWithAuxAdam. Install from source:\n"
            "    pip install git+https://github.com/KellerJordan/Muon"
        ) from exc
    from sentence_transformers import SentenceTransformer

    from lara.finetune.train import split_param_groups

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
        # dynamic=True for the same reason as the corpus embedder: length-sorted batches
        # still present many shapes, and static compilation re-autotunes on each one and
        # loses more than it gains. Measured 2.19x on this model in lara/index/embed.py.
        model[0].auto_model = torch.compile(inner, mode=rec.compile_mode, dynamic=True)

    class _Cfg:
        lr_muon, lr_adam, weight_decay = rec.lr_muon, rec.lr_adam, rec.weight_decay
    opt = SingleDeviceMuonWithAuxAdam(split_param_groups(model, _Cfg()))

    rng = random.Random(rec.seed)
    epoch_batches = query_disjoint_batches(triples, rec.batch_size, rng)
    steps = max(1, len(epoch_batches) * rec.epochs)
    warmup = max(1, int(steps * rec.warmup_frac))
    base = [g["lr"] for g in opt.param_groups]
    micro = rec.micro_batch or rec.batch_size
    step, started = 0, time.time()
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point} if rec.ema_decay else None)
    best_loss, best_state, since_best, stopped = float("inf"), None, 0, False
    val_triples = val_triples or []

    def texts(part):
        return ([QUERY_PREFIX + t.query for t in part],
                [doc_input(t.pos_text) for t in part],
                [doc_input(t.neg_text) for t in part])

    def snapshot():
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    steps_per_epoch = max(1, len(epoch_batches))
    for epoch_idx in range(rec.epochs):
        if stopped:
            break
        epoch_step = 0
        for part in query_disjoint_batches(triples, rec.batch_size, rng):
            epoch_step += 1
            scale = (step / warmup) if step < warmup else 0.5 * (
                1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale

            chunks = length_sorted_chunks(part, micro)

            def gradcache_backward(chunks=chunks):
                """One exact full-batch gradient, accumulated into .grad. Returns the loss.

                Factored out because SAM needs two of them per step: one to find the
                worst point in the neighbourhood, one to take the gradient there.
                """
                feats, cached = [], {"q": [], "p": [], "n": []}
                with torch.no_grad():
                    for sub_part in chunks:
                        qs, ps, ns_ = texts(sub_part)
                        f = [{k: v.to(device) for k, v in model.tokenize(x).items()}
                             for x in (qs, ps, ns_)]
                        feats.append(f)
                        with dev.autocast(device):
                            for key, ff in zip(("q", "p", "n"), f):
                                cached[key].append(_encode_ids(model, ff).float())
                leaves = {k: torch.cat(v).detach().requires_grad_(True)
                          for k, v in cached.items()}
                lo = mnrl(leaves["q"], leaves["p"], leaves["n"], rec.temperature)
                lo.backward()
                gr = {k: leaves[k].grad for k in ("q", "p", "n")}
                off_ = 0
                for sub_part, f in zip(chunks, feats):
                    n_i = len(sub_part)
                    with dev.autocast(device):
                        outs = [_encode_ids(model, ff) for ff in f]
                    torch.autograd.backward(
                        outs,
                        grad_tensors=[gr[k][off_:off_ + n_i].to(outs[0].dtype)
                                      for k in ("q", "p", "n")],
                    )
                    off_ += n_i
                return float(lo.detach())

            opt.zero_grad(set_to_none=True)
            total = gradcache_backward()

            if rec.sam_rho:
                # Ascent step: move to the worst point in the rho-ball, take the gradient
                # THERE, and apply that. The perturbation is scaled by the global gradient
                # norm so rho is a radius in parameter space rather than per-tensor.
                params = [q for q in model.parameters() if q.grad is not None]
                gnorm = torch.norm(torch.stack([q.grad.norm() for q in params]))
                if float(gnorm) > 0:
                    scale = rec.sam_rho / (gnorm + 1e-12)
                    # Keep the originals rather than subtracting the perturbation back
                    # off: add_ then sub_ is not exact in float32, and the ~1e-8 residue
                    # per step is a drift nothing would ever attribute to SAM. A second
                    # copy of the parameters is ~1.2 GB against 98 GB of card.
                    orig = [q.detach().clone() for q in params]
                    with torch.no_grad():
                        for q in params:
                            q.add_(q.grad.detach() * scale)
                    opt.zero_grad(set_to_none=True)
                    total = gradcache_backward()
                    with torch.no_grad():          # restore exactly before stepping
                        for q, o in zip(params, orig):
                            q.copy_(o)
                    del orig

            torch.nn.utils.clip_grad_norm_(model.parameters(), rec.grad_clip)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(rec.ema_decay).add_(v, alpha=1 - rec.ema_decay)
            step += 1

            vl = None
            if val_triples and step % rec.eval_every == 0:
                vl = mnrl_val_loss(model, val_triples, device, rec)
                if vl < best_loss - rec.min_delta:
                    best_loss, best_state, since_best = vl, snapshot(), 0
                else:
                    since_best += 1

            if progress is not None and step % 5 == 0:
                progress.send({"step": step, "steps": steps, "loss": total,
                               "lr": opt.param_groups[0]["lr"], "val_loss": vl,
                               "best_val": best_loss if best_state is not None else None,
                               "epoch": epoch_idx + 1, "epochs": rec.epochs,
                               "epoch_step": epoch_step, "steps_per_epoch": steps_per_epoch,
                               "elapsed": time.time() - started})
            if val_triples and since_best >= rec.patience:
                if progress is not None:
                    progress.send({"step": step, "steps": steps, "early_stop": True,
                                   "loss": total, "best_val": best_loss,
                                   "lr": opt.param_groups[0]["lr"],
                                   "epoch": epoch_idx + 1, "epochs": rec.epochs,
                                   "epoch_step": epoch_step,
                                   "steps_per_epoch": steps_per_epoch,
                                   # Fractional epochs, because "stopped at 2.43 epochs"
                                   # answers whether the model had seen the data through
                                   # more than once, and "stopped at step 90" does not.
                                   "epoch_frac": step / steps_per_epoch,
                                   "elapsed": time.time() - started})
                stopped = True
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    elif ema is not None:
        # Only when early stopping never picked a checkpoint: the averaged weights are a
        # regulariser, and silently overwriting a checkpoint that was chosen on validation
        # would replace a measured decision with an unmeasured one.
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(device) for k, v in ema.items()}})
    model.eval()
    model.stopped_early = stopped
    model.best_val_loss = best_loss if best_state is not None else None
    model.steps_trained = step
    return model


def mnrl_val_loss(model, triples: list[Triple], device: str, rec: "Recipe",
                  batch: int = 128) -> float:
    """Validation InfoNCE. Fixed batch, because the loss value depends on how many
    negatives it was computed against and a ragged last batch would move the number for
    reasons that have nothing to do with the model."""
    model.eval()
    total, n = 0.0, 0
    # Same disjointness rule as training, and a fixed seed so the batches — and therefore
    # the loss value — do not move between evaluations for reasons unrelated to the model.
    #
    # The batch must not exceed the number of distinct queries available: the disjoint
    # sampler yields nothing at all when it cannot fill one, and a mean over zero batches
    # is 0.0 — a "perfect" validation loss that silently stops training on the first
    # evaluation. The inner split holds ~116 queries against a default batch of 128, so
    # this is the normal case, not an edge one.
    n_queries = len({t.query_hash for t in triples})
    batch = max(8, min(batch, n_queries))
    parts = query_disjoint_batches(triples, batch, random.Random(0))
    if not parts:
        return float("inf")            # unmeasurable, so never counts as an improvement
    with torch.no_grad():
        for part in parts:
            q = _encode(model, [QUERY_PREFIX + t.query for t in part], device, grad=False)
            p = _encode(model, [doc_input(t.pos_text) for t in part], device, grad=False)
            ng = _encode(model, [doc_input(t.neg_text) for t in part], device, grad=False)
            total += float(mnrl(q.float(), p.float(), ng.float(), rec.temperature))
            n += 1
    model.train()
    return total / max(n, 1)
