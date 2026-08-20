"""5-fold fine-tune on the expanded judgements, scored raw and whitened.

Reports a 2x2 — {base, tuned} x {raw, whitened} — because whitening and training are
independent interventions and the interesting question is whether they compose. Every
number is on the held-out fold, split by query.
"""
import json, sys, time
sys.path.insert(0, '/home/user/Desktop/Local-arxiv-Research-Assistant')
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

triples = KF.make_triples(conn, max_per_query=32)
folds = KF.split_by_query(triples, K, seed=0)
print(f"{len(triples):,} triples from {len({t.query_hash for t in triples}):,} queries "
      f"| {K} folds | muon lr {LR:.0e} | batch 512 (micro 64)", flush=True)

rec = KF.Recipe(lr_muon=LR, lr_adam=LR/5, batch_size=512, micro_batch=64,
                epochs=4, patience=3, eval_every=5)
model_name = cfg.get_in('embedding.model')
rows = []

def rep(tag, m):
    wq = f"{m['within_q']:.4f}+/-{m['within_q_std']:.3f}" if m.get('within_q') is not None else "n/a"
    print(f"    {tag:22s} pair_acc {m['pair_acc']:.4f}  pooled {m['spearman']:.4f}  "
          f"within_q {wq}  mae {m['margin_mae']:.4f}", flush=True)

for fi in range(K):
    val_idx = set(folds[fi])
    val = [triples[i] for i in val_idx]
    train = [t for i, t in enumerate(triples) if i not in val_idx]
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
            elif s["step"] % 20 == 0:
                print(f"      step {s['step']:>4}/{s['steps']} loss {s['loss']:8.4f} "
                      f"lr {s['lr']:.2e}", flush=True)
    p = prog(); next(p)
    model = KF.train_on(list(inner_tr), model_name, DEVICE, rec, progress=p,
                        val_triples=inner_val)
    a_raw = KF.evaluate(model, val, dev.resolve(DEVICE))
    a_wht = KF.evaluate(model, val, dev.resolve(DEVICE), whitener=W)
    rep("tuned raw", a_raw); rep("tuned whitened", a_wht)
    print(f"      ({time.time()-t0:.0f}s, {getattr(model,'steps_trained',0)} steps)", flush=True)
    del model; dev.empty_cache(DEVICE)
    rows.append({"fold": fi, "base_raw": b_raw, "base_whitened": b_wht,
                 "tuned_raw": a_raw, "tuned_whitened": a_wht})
    json.dump(rows, open('data/logs/finetune_kfold.json','w'), indent=1)

print("\n" + "="*78)
def agg(key, metric):
    v = [r[key][metric] for r in rows if r[key].get(metric) is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float('nan'), float('nan'))
print(f"{'':18s} {'pair_acc':>18} {'pooled rho':>18} {'within-query rho':>20}")
for key in ("base_raw","base_whitened","tuned_raw","tuned_whitened"):
    a=agg(key,'pair_acc'); s=agg(key,'spearman'); w=agg(key,'within_q')
    print(f"{key:18s} {a[0]:.4f} +/-{a[1]:.4f}  {s[0]:.4f} +/-{s[1]:.4f}  {w[0]:.4f} +/-{w[1]:.4f}")
