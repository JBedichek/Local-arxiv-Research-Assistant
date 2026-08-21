"""The ranking trainer: MultipleNegativesRankingLoss, with GradCache.

The other of the two training loops (see lara/finetune/trainers.py). This one learns from
ranking rather than from magnitude: within a batch, a query's own positive must outscore
every other passage present, which is closer to what retrieval actually needs than
matching the teacher's number.

GradCache is what makes the batch big enough for that to mean anything. The contrastive
signal grows with the number of in-batch negatives, and a batch that fits in memory in one
piece is too small, so activations are recomputed per micro-batch against cached
embeddings.
"""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

from lara import device as dev
from lara.index.prefixes import QUERY_PROMPT as QUERY_PREFIX

from lara.finetune.trainers import Recipe, _encode
from lara.finetune.triples import Triple, doc_input


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
    from lara.finetune.optim import load_muon

    SingleDeviceMuonWithAuxAdam = load_muon()
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

    # ── data-parallel, without the DDP wrapper ────────────────────────────────────
    # Each rank keeps a FULL `batch_size` of its own, so per-query negatives stay at
    # 2*batch_size-1 exactly as on one GPU; the effective batch is batch_size * world.
    # Nothing about the objective changes, the gradient estimate just averages `world`
    # independent problems.
    #
    # The DDP wrapper is deliberately not used. GradCache issues one manual
    # `torch.autograd.backward` per micro-batch, and DDP's hooks would fire a gradient
    # all-reduce on each of them — either syncing many times per step or needing no_sync
    # bookkeeping around a loop that already has enough going on. All-reducing once, after
    # the full-batch gradient exists, is the same arithmetic and far easier to verify:
    # every rank ends with identical gradients, so every rank computes an identical Muon
    # update and the replicas cannot drift.
    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if ddp else 1
    rank = torch.distributed.get_rank() if ddp else 0
    if ddp:
        # Disjoint shards, so an epoch covers the data once across all ranks rather than
        # `world` times.
        triples = triples[rank::world]

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

            if ddp:
                # Average the gradients. One all-reduce over a flat buffer rather than one
                # per tensor: a 300 M-parameter model has ~450 gradient tensors, and each
                # separate NCCL call pays its own launch and synchronisation cost for a
                # payload that is often a few kilobytes.
                #
                # Clipping happens AFTER the reduce so the norm is the global one and the
                # clip threshold means the same thing at any world size.
                grads = [prm.grad for prm in model.parameters() if prm.grad is not None]
                if grads:
                    flat = torch._utils._flatten_dense_tensors(grads)
                    torch.distributed.all_reduce(
                        flat, op=torch.distributed.ReduceOp.SUM)
                    flat /= world
                    for g, merged in zip(
                            grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                        g.copy_(merged)
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
                # Rank 0 evaluates and every rank is told the answer. Letting each rank
                # decide from its own shard would let them stop on different steps, and a
                # rank that exits the loop early leaves the others hanging in all_reduce.
                if rank == 0:
                    vl = mnrl_val_loss(model, val_triples, device, rec)
                if ddp:
                    buf = torch.tensor([vl if vl is not None else 0.0],
                                       dtype=torch.float64, device=device)
                    torch.distributed.broadcast(buf, src=0)
                    vl = float(buf.item())
                if vl < best_loss - rec.min_delta:
                    best_loss, since_best = vl, 0
                    if rank == 0:
                        best_state = snapshot()
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
    if rec.val_max_triples and len(triples) > rec.val_max_triples:
        # Deterministic subsample: the same triples every evaluation, so a change in the
        # loss is a change in the model rather than a change in the sample.
        triples = random.Random(12345).sample(list(triples), rec.val_max_triples)
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
