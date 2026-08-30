"""5-fold with all four fixes from the official-recipe comparison.

  1. MultipleNegativesRankingLoss instead of MarginMSE   (the recommended objective)
  2. documents rendered as the corpus is embedded         (train/serve format match)
  3. a true 512-way batch via GradCache                   (batch = negative supply)
  4. max_seq_length 512 everywhere                        (train/serve length match)

Same folds, same LR and same metrics as the MarginMSE run, so the only differences are
the four fixes.
"""
import json, sys, time
import numpy as np
from lara.finetune import kfold as KF
from lara.finetune import whiten as WH
from lara.index.embed import load_model
from lara.store import db
from lara import config as C
from lara import device as dev

DEVICE, K, LR = "2", 5, 3e-5
cfg = C.load(); conn = db.connect(cfg.get_path('paths.metadata_db'))
W = WH.Whitener.load(cfg.get_path('disk.root')/'vectors'/'whiten_fp16.npz')

triples = KF.make_triples(conn, max_per_query=32, contextual=True)
folds = KF.split_by_query(triples, K, seed=0)
print(f"{len(triples):,} triples / {len({t.query_hash for t in triples}):,} queries", flush=True)
print(f"doc format: {triples[0].pos_text[:88]!r}", flush=True)

rec = KF.Recipe(lr_muon=LR, lr_adam=LR/5, batch_size=512, micro_batch=64,
                max_seq_length=512, epochs=4, patience=4, eval_every=10, temperature=0.05)
model_name = cfg.get_in('embedding.model')
rows = []

def rep(tag, m):
    wq = f"{m['within_q']:.4f}+/-{m['within_q_std']:.3f}" if m.get('within_q') is not None else "n/a"
    print(f"    {tag:22s} pair_acc {m['pair_acc']:.4f}  pooled {m['spearman']:.4f}  "
          f"within_q {wq}", flush=True)

for fi in range(K):
    vi = set(folds[fi])
    val = [triples[i] for i in vi]
    train = [t for i, t in enumerate(triples) if i not in vi]
    print(f"\n[fold {fi+1}/{K}] train {len(train):,}  val {len(val):,}", flush=True)

    base = load_model(model_name, device=DEVICE, max_seq_length=rec.max_seq_length)
    b_raw = KF.evaluate(base, val, dev.resolve(DEVICE))
    b_wht = KF.evaluate(base, val, dev.resolve(DEVICE), whitener=W)
    rep("base raw", b_raw); rep("base whitened", b_wht)
    del base; dev.empty_cache(DEVICE)

    inner_tr, inner_val = KF.inner_split(train, rec.inner_val_frac, seed=1)
    t0 = time.time()
    def prog():
        while True:
            s = yield
            if s.get("early_stop"):
                print(f"      early stop step {s['step']} best val {s['best_val']:.4f}", flush=True)
            elif s["step"] % 10 == 0:
                v = s.get("val_loss")
                print(f"      step {s['step']:>4}/{s['steps']} loss {s['loss']:7.4f} "
                      f"lr {s['lr']:.2e}" + (f" val {v:7.4f}" if v is not None else "")
                      + f"  {(time.time()-t0)/60:.0f}m", flush=True)
    p = prog(); next(p)
    model = KF.train_on_mnrl(list(inner_tr), model_name, DEVICE, rec, progress=p,
                             val_triples=inner_val)
    a_raw = KF.evaluate(model, val, dev.resolve(DEVICE))
    a_wht = KF.evaluate(model, val, dev.resolve(DEVICE), whitener=W)
    rep("tuned raw", a_raw); rep("tuned whitened", a_wht)
    print(f"      ({time.time()-t0:.0f}s, {getattr(model,'steps_trained',0)} steps)", flush=True)
    del model; dev.empty_cache(DEVICE)
    rows.append({"fold": fi, "base_raw": b_raw, "base_whitened": b_wht,
                 "tuned_raw": a_raw, "tuned_whitened": a_wht})
    json.dump(rows, open('data/logs/finetune_mnrl.json','w'), indent=1)

print("\n" + "="*76, flush=True)
def agg(k, m):
    v=[r[k][m] for r in rows if r[k].get(m) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'),)*2
print(f"{'':18s} {'pair_acc':>18} {'pooled rho':>18} {'within-query rho':>20}")
for k in ("base_raw","base_whitened","tuned_raw","tuned_whitened"):
    a,s_,w = agg(k,'pair_acc'), agg(k,'spearman'), agg(k,'within_q')
    print(f"{k:18s} {a[0]:.4f} +/-{a[1]:.4f}  {s_[0]:.4f} +/-{s_[1]:.4f}  {w[0]:.4f} +/-{w[1]:.4f}")
for m in ('pair_acc','spearman','within_q'):
    d=[r['tuned_raw'][m]-r['base_raw'][m] for r in rows]
    print(f"  fine-tune effect on {m:10s}: {np.mean(d):+.4f} +/-{np.std(d):.4f} "
          f"(better in {sum(1 for x in d if x>0)}/{len(d)} folds)")
