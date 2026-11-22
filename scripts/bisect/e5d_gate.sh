#!/usr/bin/env bash
# rec-e5d iter 2 gate emits for one mid candidate (dusk, :8080 up, mid-8L). Usage: e5d_gate.sh <name>
set -uo pipefail
source ~/rec-ve/e2e-env.sh >/dev/null 2>&1; source ~/opt/rocm-env.sh >/dev/null 2>&1; unset HSA_OVERRIDE_GFX_VERSION
n=$1; H=~/rec-ve/gepa/h; B=~/rec-ve/bisect/e5d; SO=$B/$n/flint_mid.so; O=$B/gate-$n; rm -rf $O; mkdir -p $O
CK=~/rec-ve/ckpt-pp2/mid-prod; MB="--bank $HOME/rec-ve/m3prep/mid-prod.bank --off $HOME/rec-ve/m3prep/mid-prod.off.npy --ckpt $CK"
log(){ echo "$(date -u +%H:%M:%SZ) [$n] $* | busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent) 8080=$(curl -s -m1 -o /dev/null -w %{http_code} http://100.98.73.29:8080/health) MemAvail=$(awk '/MemAvailable/{printf "%.1fGiB",$2/2^20}' /proc/meminfo)" | tee -a $O/gate.log; }
log "start sha $(sha256sum $SO | cut -c1-16)"
cd $H
for k in 1 2; do $PY scripts/gepa_remote.py emit --so $SO --out $O/suite$k --workers 16 --ckpt $CK 2>/dev/null | grep "^{" | tail -1 > $O/suite$k.json; done
$PY - <<P | tee -a $O/gate.log
import json,hashlib,numpy as np,os
def md5(f):
    z=np.load(f); h=hashlib.md5()
    for k in sorted(z.files):
        if k!="impl": h.update(k.encode()); h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()
a=json.load(open("$O/suite1.json"))["npz_md5"]; b=json.load(open("$O/suite2.json"))["npz_md5"]; base=os.path.expanduser("~/rec-ve/gepa/c/iter0/emit/")
print("suite determinism (2 emits):", "PASS" if a==b else "FAIL", len(a), "| G0 vs 24f167d:", sum(a[k]==md5(base+k) for k in a), "/", len(a))
P
for k in 1 2; do $PY scripts/m3_emit_wrap512.py --so $SO --ckpt $CK --out $O/wrap$k --builds fp32kv,fp32_bf16kv,bf16kv > $O/wrap$k.json 2>$O/wrap$k.err; done
$PY - <<P | tee -a $O/gate.log
import hashlib,numpy as np
def md5(f):
    z=np.load(f); return hashlib.md5(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in ("tokens","step_logits","last_logits","step_experts"))).hexdigest()
for b in ("fp32kv","fp32_bf16kv","bf16kv"):
    print("wrap512", b, "determinism:", "PASS" if md5("$O/wrap1/wrap512_%s.npz"%b)==md5("$O/wrap2/wrap512_%s.npz"%b) else "FAIL")
P
grep -h longest $O/wrap1.json | cut -c1-200 | tee -a $O/gate.log
$PY scripts/m3_depth_gate.py --rehearsal --lib $SO $MB --out $O/d8k --case depth8192 --builds fp32kv --repeats 2 --budget-s 400 2>&1 | grep -E "GATE|DETERMINISM" | cut -c1-160 | tee -a $O/gate.log
$PY scripts/m3_depth_gate.py --rehearsal --lib $SO $MB --out $O/ppl_v1 --ppl-set engine/validation/megakernel_mid/qgate/eval_set_v1.json --ppl-builds fp32kv,bf16kv --budget-s 300 2>&1 | grep PPL | cut -c1-140 | tee -a $O/gate.log
$PY scripts/m3_depth_gate.py --rehearsal --lib $SO $MB --out $O/ppl_full --ppl-set engine/validation/megakernel_mid/qgate/eval_set_full_v1.json --ppl-builds fp32kv,bf16kv --budget-s 300 2>&1 | grep PPL | cut -c1-140 | tee -a $O/gate.log
for c in short4 chat; do $PY scripts/gepa_remote.py bench --a ~/rec-ve/gepa/k2/promo/flint_mid.so --b $SO --case $c --rounds 8 --ckpt $CK 2>/dev/null | python3 -c "import json,sys;r=json.load(sys.stdin);print('$c ABBA vs 2b98737:',r['status'],'promo',r['A']['differential_tok_s'],'cand',r['B']['differential_tok_s'],'B/A',r['speedup_B_over_A'],'drops',r['rounds_dropped_cotenant'],r['cotenant_states'])" | tee -a $O/gate.log; done
log "done"
