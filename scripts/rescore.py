"""Rescore baseline and both checkpoints in the format the corpus is actually embedded in."""
import sys, json; sys.path.insert(0,'/home/user/Desktop/Local-arxiv-Research-Assistant')
from lara.finetune import evaluate as EV
from lara.index.embed import load_model
from lara.store import db; from lara import config as C
from lara import device as dev

cfg=C.load(); conn=db.connect(cfg.get_path('paths.metadata_db'))
ROOT = cfg.get_path('disk.root')/'models'
MODELS = [("baseline (stock)", cfg.get_in('embedding.model')),
          ("tuned contextual", str(ROOT/'embeddinggemma-pairs')),
          ("tuned bare (ablation)", str(ROOT/'embeddinggemma-pairs-ablation'))]
out={}
for label, path in MODELS:
    m = load_model(path, device="2", max_seq_length=512)
    row={}
    for fmt, ctx in (("corpus", True), ("bare(old)", False)):
        r = EV.run(m, conn, n=800, contextual=ctx)
        row[fmt] = {"citation": r["citation"], "paraphrase": r["paraphrase"]}
        print(f"{label:24s} [{fmt:9s}] citation mrr {r['citation']['mrr']:.4f} "
              f"r@10 {r['citation']['r@10']:.4f} | paraphrase mrr {r['paraphrase']['mrr']:.4f} "
              f"r@10 {r['paraphrase']['r@10']:.4f}", flush=True)
    out[label]=row
    del m; dev.empty_cache("2")
json.dump(out, open('data/logs/rescore.json','w'), indent=1)

print("\n=== corpus format (what search actually uses) ===")
b=out["baseline (stock)"]["corpus"]
print(f"{'model':24s} {'citation mrr':>16} {'paraphrase mrr':>18}")
for label,_ in MODELS:
    r=out[label]["corpus"]
    dc=r['citation']['mrr']-b['citation']['mrr']; dp=r['paraphrase']['mrr']-b['paraphrase']['mrr']
    print(f"{label:24s} {r['citation']['mrr']:.4f} ({dc:+.4f})  {r['paraphrase']['mrr']:.4f} ({dp:+.4f})")
