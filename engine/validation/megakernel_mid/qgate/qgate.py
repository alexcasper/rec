"""rec-1f4 quality gate (Track B: replaces byte identity for relaxed kernels). Spec: gate_spec_v1.json; frozen
eval set: eval_set_v1.json (pinned by sha256). README.md has the definition and rationale.

  emit     (dusk GPU)  run a flint_mid .so over the frozen eval set -> npz dir
           gen items x builds {fp32kv, fp32_bf16kv, bf16kv} (rec-mk-out/1 + step_experts), ppl items x
           {fp32kv, bf16kv} as one-token launches over every prefix. Every launch <= 40 positions and is
           timed; any launch > 0.25 s aborts the run (dusk display-GPU rule).
  score    (CPU)       torch fp32 reference (spec (a) pin f0698a5: rec_mlx.torch_rocm, fp32 compute, cpu)
           -> report JSON: T1/T2 spec (a) verdicts per item, T3 token agreement + margin histograms,
           T4 teacher-forced ppl.
  compare  (CPU)       candidate report vs baseline report under gate_spec_v1.json -> PASS / FAIL per tier.

    python qgate.py emit --so flint_mid.so --ckpt <mid-prod> --out DIR [--workers 16]
    python qgate.py score --ckpt <mid-prod> --dir DIR --out report.json
    python qgate.py compare --cand cand.json --base base.json
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2]
EVALSET = HERE / "eval_set_v1.json"
SPEC = HERE / "gate_spec_v1.json"
GEN_BUILDS = {  # build -> (kernel mode, KV storage dtype = reference KV tier)
    "fp32kv": (2, "fp32"),
    "fp32_bf16kv": (1, "bf16"),
    "bf16kv": (0, "bf16"),
}
PPL_BUILDS = ("fp32kv", "bf16kv")
MAX_POSITIONS = 40
MAX_LAUNCH_S = 0.25
MARGIN_BINS = [0.0, 1e-3, 5e-3, 2e-2, 1e-1, 1.0, float("inf")]


def load_evalset(path: Path = EVALSET) -> dict:
    doc = json.loads(path.read_text())
    frozen = {"gen_items": doc["gen_items"], "ppl_items": doc["ppl_items"]}
    sha = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if sha != doc["sha256"]:
        raise SystemExit(f"eval set {path} was modified: sha256 {sha} != frozen {doc['sha256']}")
    return doc


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- emit (GPU)
def cmd_emit(a) -> int:
    sys.path.insert(0, str(ENGINE / "native" / "megakernel_hip"))
    from mid_mk_driver import (MID_NL, MID_TOPK, MID_VOCAB, assert_valid_launch_outputs, pack_mid_bank,
                               shard_sha256)

    ev = load_evalset()
    sha = shard_sha256(a.ckpt)
    if sha != ev["ckpt"]["shard_sha256"]:
        raise SystemExit(f"checkpoint sha {sha} != eval set checkpoint {ev['ckpt']['shard_sha256']}")
    a.out.mkdir(parents=True, exist_ok=True)
    lib = ctypes.CDLL(str(a.so.resolve()))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.mid_session_create.restype = vp
    lib.mid_session_create.argtypes = [vp, vp, ci]
    lib.mid_session_run_ex2.restype = ci
    lib.mid_session_run_ex2.argtypes = [vp, vp, ci, vp, ci, vp, vp, ci, ci, vp, vp]
    lib.mid_session_free.restype = None
    lib.mid_session_free.argtypes = [vp]
    bank, off, _ = pack_mid_bank(a.ckpt)
    sess = lib.mid_session_create(bank.ctypes.data, off.ctypes.data, ci(64))
    longest = [0.0]

    def launch(prompt: list[int], n_out: int, mode: int, experts: bool):
        if len(prompt) + n_out > MAX_POSITIONS:
            raise SystemExit(f"launch of {len(prompt) + n_out} positions exceeds the dusk cap {MAX_POSITIONS}")
        p = np.asarray(prompt, np.int32)
        toks = np.zeros(n_out, np.int32)
        sl = np.zeros((n_out, MID_VOCAB), np.float32)
        ll = np.zeros(MID_VOCAB, np.float32)
        ex = np.full((len(p) + n_out, MID_NL, MID_TOPK), -1, np.int32) if experts else None
        ctrl = np.zeros(4, np.uint32)
        t0 = time.perf_counter()
        rc = lib.mid_session_run_ex2(sess, p.ctypes.data, ci(len(p)), toks.ctypes.data, ci(n_out), sl.ctypes.data,
                                     ll.ctypes.data, ci(a.workers), ci(mode), ctrl.ctypes.data,
                                     ex.ctypes.data if ex is not None else None)
        dt = time.perf_counter() - t0
        assert_valid_launch_outputs(rc, ctrl, toks, ll, sl, ex)
        longest[0] = max(longest[0], dt)
        if dt > MAX_LAUNCH_S:
            raise SystemExit(f"launch took {dt:.3f} s > {MAX_LAUNCH_S} s (dusk rule) -- aborting")
        return toks, sl, ll, ex

    impl = f"{a.so.name} sha256={file_sha(a.so)} workers={a.workers}"
    parts = set(a.parts.split(","))
    for b, (mode, kv) in GEN_BUILDS.items():
        if "gen" not in parts:
            break
        for it in ev["gen_items"]:
            toks, sl, ll, ex = launch(it["prompt"], it["n_out"], mode, True)
            np.savez(a.out / f"gen__{it['id'].replace('/', '__')}__{b}.npz", schema=np.array("rec-mk-out/1"),
                     case=np.array(it["id"]), ckpt_sha256=np.array(sha), impl=np.array(f"{impl} build={b}"),
                     kv_dtype=np.array(kv), prompt=np.asarray(it["prompt"], np.int32), tokens=toks,
                     step_logits=sl, last_logits=ll, step_experts=ex)
        print(f"[{time.strftime('%H:%M:%S')}] gen {b}: {len(ev['gen_items'])} items, longest launch "
              f"{longest[0]:.3f} s", flush=True)
    for b in PPL_BUILDS:
        if "ppl" not in parts:
            break
        mode = GEN_BUILDS[b][0]
        logp, top1, rowmax = [], [], []
        for it in ev["ppl_items"]:
            t = it["tokens"]
            lp, t1, mx = [], [], []
            for i in range(1, len(t)):
                _, sl, _, _ = launch(t[:i], 1, mode, False)
                row = sl[0].astype(np.float64)
                m = row.max()
                lse = m + math.log(np.exp(row - m).sum())
                lp.append(row[t[i]] - lse)
                t1.append(int(row.argmax()))
                mx.append(float(np.abs(row).max()))
            logp.append(lp), top1.append(t1), rowmax.append(mx)
        np.savez(a.out / f"ppl__{b}.npz", impl=np.array(f"{impl} build={b}"), logp=np.asarray(logp),
                 top1=np.asarray(top1, np.int32), rowmax=np.asarray(rowmax))
        print(f"[{time.strftime('%H:%M:%S')}] ppl {b}: {len(ev['ppl_items'])} items, longest launch "
              f"{longest[0]:.3f} s", flush=True)
    lib.mid_session_free(sess)
    (a.out / "emit.json").write_text(json.dumps({"evalset_sha256": ev["sha256"], "impl": impl, "parts": sorted(parts),
                                                 "longest_launch_s": longest[0]}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------ score (CPU)
def hist(values) -> list[int]:
    return np.histogram(np.asarray(values, np.float64), bins=MARGIN_BINS)[0].tolist() if len(values) else \
        [0] * (len(MARGIN_BINS) - 1)


def cmd_score(a) -> int:
    import torch

    sys.path.insert(0, str(ENGINE / "src"))
    sys.path.insert(0, str(HERE.parent))
    import mid_mk_check as mk

    ev = load_evalset()
    emit = json.loads((a.dir / "emit.json").read_text())
    if emit["evalset_sha256"] != ev["sha256"]:
        raise SystemExit("npz dir was emitted for a different eval set")
    vocab = ev["ckpt"]["vocab"]
    torch.set_num_threads(a.threads)
    ref = mk.Reference(a.ckpt, "cpu", "fp32")
    kvmap = {"fp32": None, "bf16": torch.bfloat16}
    report = {"schema": "rec-qgate-report/1", "evalset_sha256": ev["sha256"], "impl": emit["impl"],
              "longest_launch_s": emit["longest_launch_s"], "reference": "rec_mlx.torch_rocm fp32 compute, cpu "
              f"(spec (a) pin f0698a5), torch {torch.__version__}", "builds": {}, "ppl": {}}
    t_start = time.time()
    for b, (_, kv) in GEN_BUILDS.items():
        ref.model.kv_store_dtype = kvmap[kv]
        items, tf_all, tf_dis, dis_margin, n_steps, forced_rel, dev_all = [], [], [], [], 0, [], []
        for it in ev["gen_items"]:
            d = np.load(a.dir / f"gen__{it['id'].replace('/', '__')}__{b}.npz", allow_pickle=False)
            prompt, tokens = d["prompt"].tolist(), d["tokens"].tolist()
            if prompt != it["prompt"] or len(tokens) != it["n_out"]:
                raise SystemExit(f"{it['id']}/{b}: npz prompt/n_out differ from the frozen eval set")
            r = mk.check_arrays(ref, d, prompt, tokens, kv, vocab, it["id"])
            r.pop("_ref", None)
            # T3 decomposition: routing vs arithmetic. Reference forced to the kernel's OWN routing at every
            # MoE (layer, position) -> logit error given identical routing; plus every routing deviation
            # (gap = the reference's own k-th/(k+1)-th selection-score gap there).
            dev, R = mk.routing_deviations(ref, prompt, tokens, d["step_experts"])
            rows = np.abs(d["step_logits"].astype(np.float64) - R[:-1]).max(-1) / \
                np.maximum(np.abs(R[:-1]).max(-1), 1e-6)
            rec = {"id": it["id"], "pass": bool(r["pass"]), "problems": r["problems"],
                   "last_rel": r.get("last_logits_rel"), "step_rel": r.get("step_logits_max_rel"),
                   "tokens": tokens, "near_ties": len(r.get("near_ties_accepted") or []),
                   "routing_deviations": len(dev), "routing_dev_max_gap": max((x["gap"] for x in dev), default=0.0),
                   "routing_dev_non_k_k1": sum(not x["k_k1_exchange"] for x in dev),
                   "forced_step_rel": float(rows.max())}
            forced_rel.append(rec["forced_step_rel"])
            dev_all += dev
            # teacher-forced along the kernel trajectory, reference forced to the kernel's routing
            for i, tok in enumerate(tokens):
                row = R[i]
                top2 = np.partition(row, -2)[-2:]
                best = int(row.argmax())
                tf_all.append(float(top2[1] - top2[0]))
                n_steps += 1
                if tok != best:
                    tf_dis.append(float(top2[1] - top2[0]))
                    dis_margin.append(float(row[best] - row[tok]))
            items.append(rec)
        n_pass = sum(x["pass"] for x in items)
        rels = [x["step_rel"] for x in items if x["step_rel"] is not None]
        lrels = [x["last_rel"] for x in items if x["last_rel"] is not None]
        report["builds"][b] = {
            "ref_kv": kv, "logit_rtol": mk.LOGIT_RTOL[kv], "items": len(items), "spec_a_pass": n_pass,
            "spec_a_all_pass": n_pass == len(items),
            "step_rel_max": max(rels) if rels else None, "step_rel_median": float(np.median(rels)) if rels else None,
            "last_rel_max": max(lrels) if lrels else None,
            "forced_step_rel_max": max(forced_rel), "forced_step_rel_median": float(np.median(forced_rel)),
            "routing_deviations": len(dev_all), "routing_dev_non_k_k1": sum(not x["k_k1_exchange"] for x in dev_all),
            "routing_dev_gap_max": max((x["gap"] for x in dev_all), default=0.0),
            "routing_dev_gap_hist": np.histogram([x["gap"] for x in dev_all],
                                                 bins=[0, 1e-5, 1e-4, 2.5e-4, 1e-3, 1e-2, np.inf])[0].tolist(),
            "tf_reference": "fp32 reference forced to the kernel's own routing (step_experts)",
            "tf_steps": n_steps, "tf_disagreements": len(dis_margin),
            "tf_agreement": 1.0 - len(dis_margin) / max(n_steps, 1),
            "tf_hard_disagreements": sum(m > mk.TIE_TOL for m in dis_margin),
            "tf_disagreement_margin_max": max(dis_margin) if dis_margin else 0.0,
            "margin_bins": MARGIN_BINS[:-1] + ["inf"],
            "ref_top12_margin_hist_all": hist(tf_all), "ref_top12_margin_hist_at_disagreement": hist(tf_dis),
            "per_item": items,
        }
        rb = report["builds"][b]
        print(f"[{time.time() - t_start:6.0f}s] {b}: spec(a) {n_pass}/{len(items)}, step_rel_max "
              f"{rb['step_rel_max']:.3e}, forced-routing step_rel_max {rb['forced_step_rel_max']:.3e}, routing "
              f"deviations {rb['routing_deviations']} (non-k/k+1 {rb['routing_dev_non_k_k1']}, max gap "
              f"{rb['routing_dev_gap_max']:.2e}), tf disagreements {len(dis_margin)}/{n_steps} "
              f"(hard {rb['tf_hard_disagreements']})", flush=True)
    # T4: teacher-forced ppl; reference at the build's KV tier
    ref_lp = {}
    for kv in ("fp32", "bf16"):
        ref.model.kv_store_dtype = kvmap[kv]
        rows = []
        with torch.inference_mode():
            for it in ev["ppl_items"]:
                t = it["tokens"]
                lg = ref.model.forward(torch.tensor([t]))[0].double()
                lp = torch.log_softmax(lg, -1)[:-1].gather(1, torch.tensor(t[1:]).view(-1, 1))[:, 0]
                rows.append(lp.numpy())
        ref_lp[kv] = np.asarray(rows)
    for b in PPL_BUILDS:
        kv = GEN_BUILDS[b][1]
        p = np.load(a.dir / f"ppl__{b}.npz")
        klp, rlp = p["logp"], ref_lp[kv]
        nll_k, nll_r = float(-klp.mean()), float(-rlp.mean())
        report["ppl"][b] = {"ref_kv": kv, "n_tokens": int(klp.size), "nll": nll_k, "ppl": math.exp(nll_k),
                            "ref_nll": nll_r, "ref_ppl": math.exp(nll_r),
                            "ppl_rel_vs_ref": math.exp(nll_k) / math.exp(nll_r) - 1.0,
                            "max_abs_dlogp": float(np.abs(klp - rlp).max()),
                            "per_item_nll": (-klp.mean(1)).tolist(), "ref_per_item_nll": (-rlp.mean(1)).tolist()}
        print(f"ppl {b}: kernel {math.exp(nll_k):.6g} ref {math.exp(nll_r):.6g} rel "
              f"{report['ppl'][b]['ppl_rel_vs_ref']:+.3e} max|dlogp| {report['ppl'][b]['max_abs_dlogp']:.3e}")
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t_start:.0f} s)")
    return 0


# ---------------------------------------------------------------------------------------- compare (CPU)
def cmd_compare(a) -> int:
    spec = json.loads(SPEC.read_text())
    c, bse = json.loads(a.cand.read_text()), json.loads(a.base.read_text())
    ev = load_evalset()
    out, ok = [], True

    def gate(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        out.append({"gate": name, "pass": bool(passed), "detail": detail})

    gate("evalset", c["evalset_sha256"] == bse["evalset_sha256"] == ev["sha256"] == spec["evalset_sha256"],
         "candidate, baseline, spec and frozen file name the same eval set")
    cb = c["builds"]["fp32kv"]
    fails = [x["id"] for x in cb["per_item"] if not x["pass"]]
    gate("T1 spec(a) fp32kv (absolute)", cb["spec_a_all_pass"],
         f"{cb['spec_a_pass']}/{cb['items']} pass, step_rel_max {cb['step_rel_max']:.3e} (rtol {cb['logit_rtol']})"
         + (f"; failing {fails[:6]}" if fails else ""))
    t2 = spec["T2_fp32_bf16kv"]
    cb, bb = c["builds"]["fp32_bf16kv"], bse["builds"]["fp32_bf16kv"]
    regress = []
    for x, y in zip(cb["per_item"], bb["per_item"]):
        if y["pass"] and not x["pass"]:
            regress.append(f"{x['id']} (new fail: {'; '.join(x['problems'])[:80]})")
        elif not y["pass"] and not x["pass"] and (x["step_rel"] > t2["baseline_fail_rel_factor"] * y["step_rel"]
                                                    or len(x["problems"]) > len(y["problems"])):
            regress.append(f"{x['id']} (baseline-failing item got worse: {x['step_rel']:.3e} vs {y['step_rel']:.3e})")
    gate("T2 spec(a) fp32_bf16kv (no per-item regression)", not regress,
         f"candidate {cb['spec_a_pass']}/{cb['items']}, baseline {bb['spec_a_pass']}/{bb['items']}"
         + (f"; regressions {regress[:4]}" if regress else ""))
    t3 = spec["T3_bf16kv_production"]
    cb, bb = c["builds"]["bf16kv"], bse["builds"]["bf16kv"]
    lim_dev = math.ceil(t3["routing_dev_factor"] * bb["routing_deviations"]) + t3["routing_dev_slack"]
    gate("T3 routing deviations", cb["routing_deviations"] <= lim_dev,
         f"candidate {cb['routing_deviations']} vs limit {lim_dev} (baseline {bb['routing_deviations']})")
    lim_gap = max(t3["routing_gap_factor"] * bb["routing_dev_gap_max"], t3["routing_gap_floor"])
    gate("T3 routing deviation max gap", cb["routing_dev_gap_max"] <= lim_gap,
         f"candidate {cb['routing_dev_gap_max']:.3e} vs limit {lim_gap:.3e} (baseline {bb['routing_dev_gap_max']:.3e})")
    gate("T3 forced-routing logit rel err", cb["forced_step_rel_max"] <= t3["step_rel_max_factor"] * bb["forced_step_rel_max"],
         f"candidate {cb['forced_step_rel_max']:.3e} vs {t3['step_rel_max_factor']} x baseline {bb['forced_step_rel_max']:.3e}")
    gate("T3 forced-routing hard disagreements",
         cb["tf_hard_disagreements"] <= bb["tf_hard_disagreements"] + t3["hard_disagreements_slack"],
         f"candidate {cb['tf_hard_disagreements']} vs baseline {bb['tf_hard_disagreements']} "
         f"(margin > {t3['hard_disagreement_margin']}, slack {t3['hard_disagreements_slack']})")
    gate("T3 forced-routing token agreement", cb["tf_agreement"] >= bb["tf_agreement"] - t3["agreement_drop_max"],
         f"candidate {cb['tf_agreement']:.4f} vs baseline {bb['tf_agreement']:.4f} (max drop {t3['agreement_drop_max']})")
    t4 = spec["T4_ppl"]
    cp, bp = c["ppl"]["fp32kv"], bse["ppl"]["fp32kv"]
    gate("T4 ppl fp32kv vs torch", abs(cp["ppl_rel_vs_ref"]) <= t4["fp32kv_rel_vs_ref_max"],
         f"{cp['ppl_rel_vs_ref']:+.3e} (max {t4['fp32kv_rel_vs_ref_max']}; baseline {bp['ppl_rel_vs_ref']:+.3e})")
    cp, bp = c["ppl"]["bf16kv"], bse["ppl"]["bf16kv"]
    lim = max(t4["bf16kv_rel_factor"] * abs(bp["ppl_rel_vs_ref"]), t4["bf16kv_rel_floor"])
    gate("T4 ppl bf16kv vs torch", abs(cp["ppl_rel_vs_ref"]) <= lim,
         f"{cp['ppl_rel_vs_ref']:+.3e} (max {lim:.3e} = max({t4['bf16kv_rel_factor']} x baseline "
         f"{abs(bp['ppl_rel_vs_ref']):.3e}, floor {t4['bf16kv_rel_floor']}))")
    info = {}
    for b in GEN_BUILDS:  # informational: free-running divergence vs the baseline kernel's own trajectory
        fd = []
        for x, y in zip(c["builds"][b]["per_item"], bse["builds"][b]["per_item"]):
            diff = [i for i, (p, q) in enumerate(zip(x["tokens"], y["tokens"])) if p != q]
            fd.append(diff[0] if diff else None)
        info[b] = {"items_identical": sum(f is None for f in fd), "items": len(fd),
                   "first_divergence_steps": [f for f in fd if f is not None]}
    res = {"verdict": "PASS" if ok else "FAIL", "spec_version": spec["version"], "candidate": c["impl"],
           "baseline": bse["impl"], "gates": out, "freerun_vs_baseline": info}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--so", type=Path, required=True)
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--parts", default="gen,ppl", help="gen,ppl (default) | ppl | gen")
    s = sub.add_parser("score")
    s.add_argument("--ckpt", type=Path, required=True)
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--threads", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("--cand", type=Path, required=True)
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    return {"emit": cmd_emit, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
