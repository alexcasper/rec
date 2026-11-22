"""rec-1f4 gate self-test: build a negatively-controlled copy of an emit dir with KNOWN defects, so that
`qgate.py score` + `compare` against the clean report must FAIL exactly the targeted gates:

  T1  fp32kv  item chat/03 : one non-argmax logit in step row 5 moved by +3e-4 x row absmax (rel 3e-4 > 1e-4)
  T3  bf16kv  item plain/04: every step row scaled by (1 + 2e-2)  (forced-routing rel >> 1.5 x baseline)
  T4  fp32kv  ppl          : every teacher-forced logprob + 2e-4  (ppl rel ~ -2e-4, limit 1e-4)
Everything else is a hard link of the clean file (T2 must still PASS).

    python selftest_negative.py CLEAN_DIR NC_DIR
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=False)
for f in src.iterdir():
    os.link(f, dst / f.name)


def rewrite(name, fn):
    p = dst / name
    d = dict(np.load(p))
    os.unlink(p)
    fn(d)
    np.savez(p, **d)


def t1(d):
    row = d["step_logits"][5]
    j = int(row.argmin())
    row[j] += 3e-4 * np.abs(row).max()


def t3(d):
    d["step_logits"] = (d["step_logits"] * (1 + 2e-2)).astype(np.float32)


def t4(d):
    d["logp"] = d["logp"] + 2e-4


rewrite("gen__chat__03__fp32kv.npz", t1)
rewrite("gen__plain__04__bf16kv.npz", t3)
rewrite("ppl__fp32kv.npz", t4)
shutil.copy(src / "emit.json", dst / "emit.json.tmp")
os.replace(dst / "emit.json.tmp", dst / "emit.json")
print(f"negative-control dir {dst}: T1 (chat/03 fp32kv), T3 (plain/04 bf16kv), T4 (ppl fp32kv) perturbed")
