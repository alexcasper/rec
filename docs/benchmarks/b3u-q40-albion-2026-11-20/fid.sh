#!/usr/bin/env bash
# rec-b3u fidelity: llama.cpp HIP (a37cc46, R9700) greedy vs torch fp32 CPU reference, teacher-forced
# (step-0 mid_gguf_fidelity.py, adapted only in its llama.cpp backend). Usage: fid.sh GGUF CKPT OUT.json
set -u
cd ~/rec-ve/b3u
GGUF=$1; CK=$2; OUT=$3; PORT=18181
export LD_LIBRARY_PATH=/opt/rocm/lib:$(ls -d /nix/store/*-gcc-14.2.1.20250322-lib | head -1)/lib:$(ls -d /nix/store/*-zstd-1.5.7 | head -1)/lib:$(ls -d /nix/store/*-zlib-1.3.2 | head -1)/lib
unset HSA_OVERRIDE_GFX_VERSION; export HIP_VISIBLE_DEVICES=0
llama.cpp/build-hip/bin/llama-server -m $GGUF -ngl 99 -fa on -c 4096 -np 1 --host 127.0.0.1 --port $PORT > ${OUT%.json}.server.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 150); do curl -s -m2 127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m2 127.0.0.1:$PORT/health; echo
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" timeout 3000 ~/rec-ve/.venv/bin/python mid_gguf_fidelity.py ~/rec-ve/u9q/gk/engine/src $CK $PORT > $OUT 2> ${OUT%.json}.err
echo "fidelity exit $?"
~/rec-ve/.venv/bin/python -c "
import json; d=json.load(open('$OUT'))
for k, v in d.items(): print(k, 'agree %d/%d' % (v['agree'], v['n']), 'distinct', v['distinct'], 'flips', [(f['step'], round(f['margin'], 4)) for f in v['flips']])"
