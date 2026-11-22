# rec-1f4 quality gate v1: re-baseline + self-test records

Gate definition: `engine/validation/megakernel_mid/qgate/` (README, gate_spec_v1.json). Frozen eval set:
eval_set_v1.json, sha256 72209796...f464, commit 16b2c42.

- `emit_baseline.log`: dusk emit of 2fd83cb (s4/base) and 24f167d (s5/b7). Cotenants logged per line
  (flint idle, E4B idle, qwen inactive); longest launch 0.225 s / 0.197 s; no kernel ring errors.
  24f167d vs 2fd83cb: 98/98 npz byte-identical (checked on dusk), so one report serves both.
- `report_base_2fd83cb.json`: THE re-baseline (Step 1 reference for every relaxed candidate). CPU
  reference on dawn, torch 2.15.0, 12 threads, 632 s.
- `compare_self.json`: baseline vs itself, PASS (10/10 gates).
- `report_nc_base.json` + `compare_negative.json`: negative control (`selftest_negative.py`). FAIL on exactly
  T1 / T3 forced-rel / T4 fp32kv; the other 7 gates pass.
