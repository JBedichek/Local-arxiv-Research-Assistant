"""Fine-tune the embedder on citation structure with Muon (PLAN.md §5.2).

Training signal — bag-level, chunk-to-chunk::

    bag       one citation edge A -> B
    instances m body chunks sampled from A, m from B
    positive  the (B,B) diagonal after LogSumExp-pooling the m x m chunk similarities
    negative  every other paper in the batch, pooled the same way

**Why not the obvious construction.** Anchoring on the paragraph that contains the
citation and pairing it with the cited abstract looks natural and is measurably biased:
across 131k extracted contexts, 90% sit in Introduction, Related Work or Background. That
pairing teaches "related-works prose -> abstract", a register the product never
encounters — readers highlight method paragraphs and ask about ablations, and we search
body chunks. Both sides are ordinary chunks here, so the training distribution matches the
serving distribution at both ends, and nothing longer than a chunk is ever encoded.

**Muon applies to 2D matrices only.** It orthogonalises the momentum update via
Newton-Schulz, which is meaningful for a weight matrix and meaningless for a bias, a
LayerNorm gain, or an embedding table — those keep AdamW. ``SingleDeviceMuonWithAuxAdam``
takes both kinds as separate parameter groups, so the split is declared, not implied.
"""

from __future__ import annotations

import math
import random
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    model: str = "google/embeddinggemma-300m"
    device: str = "cuda:0"
    # Chunks average 219 tokens (p95 432). Training at 512 pads most of the batch with
    # nothing while paying full attention cost for it.
    max_seq_length: int = 384
    amp_dtype: str = "bfloat16"    # activations in bf16; master weights stay fp32
    grad_checkpointing: bool = True
    batch_size: int = 16           # bags (citation edges) per step
    chunks_per_paper: int = 4      # m: chunks sampled from each side of a bag
    epochs: int = 1
    lr_muon: float = 2e-3          # Muon tolerates a far larger LR than Adam on matrices
    lr_adam: float = 2e-5          # embeddings/norms stay conservative
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    temperature: float = 0.05      # InfoNCE scale; 0.05 == logits x20
    max_edges: int = 200_000       # 0 = all usable edges
    min_chunks: int = 8            # papers with fewer are too thin to sample from
    pool_temp: float = 0.05        # LogSumExp sharpness over candidate chunk pairs
    compile_mode: str | None = "default"
    eval_every: int = 0            # steps; 0 = only before/after
    seed: int = 0
    held_out_frac: float = 0.2
    grad_clip: float = 1.0
    stats: dict = field(default_factory=dict)


DOC_PREFIX = "title: {title} | text: "
QUERY_PREFIX = "task: search result | query: "


def mil_nce(src: torch.Tensor, dst: torch.Tensor, temperature: float,
            pool_temp: float = 0.05) -> torch.Tensor:
    """Bag-level InfoNCE over chunk pairs.

    ``src``/``dst`` are (B, m, D): B citation edges, m chunks sampled from each paper.

    For every ordered pair of bags we pool the m x m chunk similarities with LogSumExp,
    giving a (B, B) matrix whose diagonal is the cited pair. LogSumExp is a soft maximum:
    it concentrates on the best-matching chunk pair without routing the entire gradient
    through it, which matters because a hard argmax would only ever reinforce the
    alignment the current model already prefers.
    """
    a = F.normalize(src, dim=-1)
    b = F.normalize(dst, dim=-1)
    # (B, B, m, n) — every bag against every bag, every chunk against every chunk
    sim = torch.einsum("imd,jnd->ijmn", a, b)
    B, _, m, n = sim.shape
    pooled = torch.logsumexp(sim.reshape(B, B, m * n) / pool_temp, dim=-1) * pool_temp

    logits = pooled / temperature
    target = torch.arange(B, device=src.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def margin_mse(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
               teacher_margin: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Distil the teacher's MARGIN, not its verdict.

    The cross-encoder does not merely say relevant/irrelevant; it says how much. Rounding
    that to a binary label throws away most of the supervision — a pair at 0.95 vs 0.05
    and a pair at 0.55 vs 0.45 become identical training signal despite meaning very
    different things.

    MarginMSE asks the student to reproduce the teacher's gap rather than to classify,
    which also avoids the pathology of pushing every positive to cosine 1.0 and collapsing
    the representation.
    """
    qn = F.normalize(q, dim=-1)
    student_margin = (qn * F.normalize(pos, dim=-1)).sum(-1) \
                   - (qn * F.normalize(neg, dim=-1)).sum(-1)
    return F.mse_loss(student_margin * scale, teacher_margin * scale)


def sample_bag(bag, m: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """m chunks from each side, sampled fresh every epoch so a bag is not memorised."""
    src = rng.sample(bag.src_chunks, min(m, len(bag.src_chunks)))
    dst = rng.sample(bag.dst_chunks, min(m, len(bag.dst_chunks)))
    while len(src) < m:
        src.append(rng.choice(bag.src_chunks))
    while len(dst) < m:
        dst.append(rng.choice(bag.dst_chunks))
    return src, dst


def split_param_groups(model, cfg: TrainConfig) -> list[dict]:
    """2D weight matrices to Muon; everything else to AdamW."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2 and "embed" not in name.lower()
        (muon_params if is_matrix else adam_params).append(p)
    return [
        {"params": muon_params, "use_muon": True,
         "lr": cfg.lr_muon, "weight_decay": cfg.weight_decay, "momentum": 0.95},
        {"params": adam_params, "use_muon": False,
         "lr": cfg.lr_adam, "weight_decay": cfg.weight_decay, "betas": (0.9, 0.95),
         "eps": 1e-10},
    ]


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, weights: torch.Tensor,
             temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE over in-batch negatives."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = (a @ p.T) / temperature
    target = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits, target, reduction="none")
    loss_p = F.cross_entropy(logits.T, target, reduction="none")
    return ((loss_a + loss_p) * 0.5 * weights).sum() / weights.sum().clamp(min=1e-6)


def encode_batch(model, texts: list[str], device: str) -> torch.Tensor:
    """Tokenise and encode with gradients attached.

    sentence-transformers' ``encode()`` runs under no_grad, so training has to go through
    the module stack directly.
    """
    features = model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items()}
    out = model(features)
    return out["sentence_embedding"]


def train(conn: sqlite3.Connection, cfg: TrainConfig, progress=None):
    from muon import SingleDeviceMuonWithAuxAdam
    from sentence_transformers import SentenceTransformer

    from lara.finetune import evaluate as EV

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    held = EV._held_out_srcs(conn, cfg.held_out_frac, cfg.seed)
    model = SentenceTransformer(cfg.model, device=cfg.device,
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = cfg.max_seq_length
    model.train()

    # Activation memory, not parameters, is what makes this run large: 192 sequences of
    # 512 tokens through 24 layers in fp32 with everything retained for backward needed
    # 94 GiB and OOMed a 96 GiB card. Checkpointing recomputes activations in the backward
    # pass instead of storing them, and bf16 autocast halves what is stored either way.
    if cfg.grad_checkpointing:
        inner = model[0].auto_model
        if hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        inner.config.use_cache = False

    from lara.finetune import bags as BG

    all_bags = BG.build_bags(conn, min_chunks=cfg.min_chunks,
                             per_paper=max(8, cfg.chunks_per_paper * 3),
                             limit_edges=cfg.max_edges)
    # Hold out by CITING PAPER, matching the eval split, so a paper never appears in both.
    train_bags = [b for b in all_bags if b.src not in held and b.dst not in held]
    if len(train_bags) < cfg.batch_size * 4:
        raise RuntimeError(f"only {len(train_bags)} usable bags; crawl more papers first")

    if cfg.compile_mode:
        # dynamic=True for the same reason as the indexing run: length-sorted batches give
        # the encoder many shapes, and static compilation re-tunes on each of them.
        model[0].auto_model = torch.compile(
            model[0].auto_model, mode=cfg.compile_mode, dynamic=True
        )

    opt = SingleDeviceMuonWithAuxAdam(split_param_groups(model, cfg))
    steps = (len(train_bags) // cfg.batch_size) * cfg.epochs
    warmup = max(1, int(steps * cfg.warmup_frac))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int) -> None:
        if step < warmup:
            scale = step / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale

    amp = getattr(torch, cfg.amp_dtype)
    rng = random.Random(cfg.seed)
    history: list[dict] = []
    step = 0
    started = time.time()
    m = cfg.chunks_per_paper

    for epoch in range(cfg.epochs):
        rng.shuffle(train_bags)
        for i in range(0, len(train_bags) - cfg.batch_size + 1, cfg.batch_size):
            batch = train_bags[i : i + cfg.batch_size]
            set_lr(step)

            src_texts, dst_texts = [], []
            for bag in batch:
                s_c, d_c = sample_bag(bag, m, rng)
                src_texts.extend(s_c)
                dst_texts.extend(d_c)

            # One forward pass for both sides keeps the batch shapes stable for compile.
            with torch.autocast(device_type="cuda", dtype=amp):
                emb = encode_batch(model, src_texts + dst_texts, cfg.device)
                D = emb.size(-1)
                half = len(src_texts)
                src = emb[:half].view(len(batch), m, D)
                dst = emb[half:].view(len(batch), m, D)
                # Loss in fp32: the (B,B,m,n) LogSumExp is numerically delicate at bf16.
                loss = mil_nce(src.float(), dst.float(), cfg.temperature, cfg.pool_temp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step += 1
            if progress is not None and step % 10 == 0:
                progress.send({"step": step, "steps": steps, "epoch": epoch,
                               "loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"],
                               "elapsed": time.time() - started})
            if cfg.eval_every and step % cfg.eval_every == 0:
                model.eval()
                history.append({"step": step, **EV.run(model, conn, n=200, pool_size=800)})
                model.train()

    model.eval()
    return model, {"steps": step, "bags": len(train_bags), "held_out_papers": len(held),
                   "seconds": time.time() - started, "history": history}
