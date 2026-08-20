"""Whitening the embedding space, fitted on the corpus we already embedded.

EmbeddingGemma's output space on this corpus is severely anisotropic. Measured over
400,000 corpus vectors:

    condition number   5,624,824
    effective rank         116.6  of 768

So the 768 dimensions carry about 117 dimensions worth of variance. Cosine similarity
treats every axis as equally informative, which is exactly wrong in a space shaped like
that: the handful of high-variance directions dominate every comparison, and the
directions that actually separate one ML paper from another are compressed into the
noise floor.

Whitening removes the mean and rescales the axes to unit variance, so no direction can
dominate by amplitude alone. It is post-hoc and costs one 768x768 matrix — no
re-embedding, no retraining. Measured on held-out judgement triples it moved pairwise
accuracy 0.8898 -> 0.9220 with the shrinkage below.

**This changes ordering, which is the point and also the risk.** Whitening is not a
monotone transform of cosine similarity, so ranks move. It improved the binary decision
(is this passage the relevant one) while *lowering* correlation with the teacher's margin
magnitudes — which is expected, since rescaling the geometry rescales the gaps, and the
teacher's scale is not something retrieval needs reproduced.

**Shrinkage is not optional.** The smallest eigenvalues are ~1e-8, and dividing by their
square roots amplifies directions that are pure estimation noise by four orders of
magnitude. Blending the covariance toward a scaled identity bounds that. At shrink=0.0
pairwise accuracy was 0.9181; at 0.1 it was 0.9220, and the difference is entirely in how
much noise the inverse square root is allowed to amplify.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Fraction of the mean eigenvalue blended into every eigenvalue before inversion.
#: Swept over {0, 0.01, 0.1, 1.0} on held-out triples; 0.1 won.
DEFAULT_SHRINK = 0.1


class Whitener:
    """Mean removal plus covariance equalisation, as one linear map."""

    def __init__(self, mean: np.ndarray, W: np.ndarray, meta: dict | None = None) -> None:
        self.mean = mean.astype(np.float32)
        self.W = W.astype(np.float32)
        self.meta = meta or {}

    def __call__(self, X: np.ndarray, renormalize: bool = True) -> np.ndarray:
        """Whiten a batch of row vectors, renormalising so cosine stays meaningful."""
        Y = (np.asarray(X, dtype=np.float32) - self.mean) @ self.W
        if renormalize:
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, W=self.W, **{
            f"meta_{k}": np.asarray(v) for k, v in self.meta.items()
        })

    @classmethod
    def load(cls, path: str | Path) -> "Whitener":
        d = np.load(Path(path))
        meta = {k[5:]: d[k].item() for k in d.files if k.startswith("meta_")}
        return cls(d["mean"], d["W"], meta)


def fit(vectors_path: str | Path, *, dim: int = 768, sample: int = 400_000,
        shrink: float = DEFAULT_SHRINK, seed: int = 0) -> Whitener:
    """Estimate the transform from a random sample of the corpus vectors.

    Sampled rather than exhaustive: a 768x768 covariance from 400 k rows is already far
    past the point where more rows move the estimate, and reading all 29.6 M would cost
    45 GB of I/O to refine a number in the fourth decimal place.

    Rows are sampled from the whole file including the orphan tail (see
    lara/index/vectors.py) — orphans are real embeddings of real chunks that were merely
    re-embedded elsewhere, so they carry the same distribution and excluding them would
    cost a join for no statistical gain.
    """
    X = np.memmap(Path(vectors_path), dtype=np.float16, mode="r").reshape(-1, dim)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    take = min(sample, n)
    idx = np.sort(rng.choice(n, take, replace=False))
    S = np.asarray(X[idx], dtype=np.float32)

    mean = S.mean(axis=0)
    Sc = S - mean
    cov = (Sc.T @ Sc) / max(len(Sc) - 1, 1)
    ev, V = np.linalg.eigh(cov)
    ev = np.maximum(ev, 0.0)

    eff_rank = float((ev.sum() ** 2) / max(float((ev ** 2).sum()), 1e-30))
    cond = float(ev.max() / max(ev.min(), 1e-12))

    # Cov^{-1/2}, with the shrinkage that stops near-zero eigenvalues exploding.
    e = ev + shrink * float(ev.mean())
    W = (V / np.sqrt(e)) @ V.T

    return Whitener(mean, W, {
        "sample": int(take), "dim": int(dim), "shrink": float(shrink),
        "effective_rank": round(eff_rank, 2), "condition": round(cond, 1),
    })
