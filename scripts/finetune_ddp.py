"""Data-parallel fine-tune across all GPUs, launched with torchrun.

    torchrun --nproc_per_node=3 scripts/finetune_ddp.py --epochs 1

Each rank holds a full `batch_size`, so the effective batch is batch_size * world and the
per-query negative set is unchanged. Gradients are averaged once per step, after GradCache
has produced the full-batch gradient on each rank.

Rank 0 owns everything that is not training: the independent eval, the checkpoint, and the
guard. The others only contribute gradients.
"""
import argparse, os, sys, time
sys.path.insert(0, '/home/user/Desktop/Local-arxiv-Research-Assistant')
import torch
import torch.distributed as dist

from lara.finetune import evaluate as EV
from lara.finetune import kfold as KF
from lara.index.embed import load_model
from lara.store import db
from lara import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr-muon", type=float, default=3e-5)
ap.add_argument("--batch-size", type=int, default=512, help="PER RANK")
ap.add_argument("--micro-batch", type=int, default=64)
ap.add_argument("--max-per-query", type=int, default=32)
ap.add_argument("--max-seq-length", type=int, default=512)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--n-eval", type=int, default=800)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--contextual", action="store_true", default=False)
ap.add_argument("--out", default="data/models/embeddinggemma-ddp")
a = ap.parse_args()

local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
device = f"cuda:{local}"
log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

cfg = C.load(); conn = db.connect(cfg.get_path('paths.metadata_db'))
model_name = cfg.get_in('embedding.model')
triples = KF.make_triples(conn, max_per_query=a.max_per_query, contextual=a.contextual)
train, val = KF.inner_split(triples, 0.15, seed=1)
log(f"{len(triples):,} triples / {len({t.query_hash for t in triples}):,} queries")
log(f"world {world} x batch {a.batch_size} = effective {a.batch_size*world} · "
    f"{a.epochs} epoch(s) · seq {a.max_seq_length}")

rec = KF.Recipe(lr_muon=a.lr_muon, lr_adam=a.lr_muon/5, batch_size=a.batch_size,
                micro_batch=a.micro_batch, epochs=a.epochs,
                max_seq_length=a.max_seq_length, patience=a.patience,
                eval_every=a.eval_every, seed=a.seed, compile_mode="default")

before = None
if rank == 0:
    base = load_model(model_name, device=device, max_seq_length=a.max_seq_length)
    before = EV.run(base, conn, n=a.n_eval)
    log("\nbaseline on the independent eval"); log(EV.format_report(before))
    del base
    torch.cuda.empty_cache()
dist.barrier()

def prog():
    while True:
        s = yield
        if s.get("early_stop"):
            log(f"  early stop at step {s['step']}/{s['steps']} — epoch "
                f"{s.get('epoch')}/{s.get('epochs')}, {s.get('epoch_frac',0):.2f} epochs "
                f"· best val {s['best_val']:.4f}")
        elif s["step"] % 5 == 0:
            v = s.get("val_loss")
            log(f"  step {s['step']:>4}/{s['steps']}  ep {s['step']/max(1,s.get('steps_per_epoch',1)):5.2f}"
                f"  loss {s['loss']:7.4f}  lr {s['lr']:.2e}"
                + (f"  val {v:7.4f}" if v is not None else "")
                + f"  {s['elapsed']/60:.0f}m")

p = prog(); next(p)
t0 = time.time()
model = KF.train_on_mnrl(list(train), model_name, device, rec,
                         progress=(p if rank == 0 else None), val_triples=val)
dist.barrier()
log(f"trained {getattr(model,'steps_trained',0)} steps in {(time.time()-t0)/60:.1f} min")

if rank == 0:
    after = EV.run(model, conn, n=a.n_eval)
    log("\nafter training"); log(EV.format_report(before, after))
    won = after["citation"]["mrr"] > before["citation"]["mrr"]
    kept = after["paraphrase"]["mrr"] > before["paraphrase"]["mrr"] - 0.02
    log(f"\nguard: citation {'PASS' if won else 'FAIL'} · "
        f"paraphrase {'PASS' if kept else 'FAIL'}")
    model.save(a.out)
    log(f"saved {a.out}")
dist.barrier()
dist.destroy_process_group()
