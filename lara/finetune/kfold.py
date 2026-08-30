"""Compatibility shim: the old monolithic kfold module.

The kfold code was split into `triples.py` (data/folds), `trainers.py`
(MarginMSE recipe/train/eval) and `mnrl.py` (MultipleNegativesRankingLoss).
The scripts under scripts/ still import this module as `KF`; re-export the
public names they use so those entry points keep working.

Fix for bug reference C6 (2026-08-30).
"""
from lara.finetune.triples import (  # noqa: F401
    Triple,
    make_triples,
    split_by_query,
    inner_split,
)
from lara.finetune.trainers import (  # noqa: F401
    Recipe,
    evaluate,
    train_on,
    validation_loss,
    sweep,
    format_sweep,
)
from lara.finetune.mnrl import (  # noqa: F401
    train_on_mnrl,
    mnrl_val_loss,
    query_disjoint_batches,
)
