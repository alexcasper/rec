# Phase profile of run-of-record 24f167d, dusk mid-8L (rec-ohs, 2026-11-22)

Profiler: rec-6ob `mk_prof.py` + MK_PROFILE instrumentation (`mk_profile_24f167d.patch`, applied to 24f167d's
kernel). Worker 0 / thread 0 timestamps every SYNC_ALL. One launch = prompt4 + 36 decode (0.195 s, under the 0.25 s
cap). Two independent runs x 5 reps, each behind busy<5 x3 + wait-for-idle. Cotenant state before and after each
run: `8080:idle 8090:idle qwen:inactive` (unchanged, so comparable). Instrumentation overhead +2 % launch time;
tokens identical to the production 24f167d .so.

Token: 4860 / 4850 us (205.8 / 206.2 tok/s in-kernel; ABBA harness 207-209). Roof = ~82 GB/s.

| phase (per token) | n | us | MB | GB/s | % roof | us @82 | loss us |
|---|---|---|---|---|---|---|---|
| O-proj (K=4096, N=2048, generic pipelined path) | 8 | 674 | 37.7 | **56.0** | 68 % | 460 | **214** |
| MoE down (2 x K=512 VCgen + K=1024 generic, N=2048) | 7 | 365 | 16.5 | **45.3** | 55 % | 201 | **164** |
| QKV (K=2048 segmented) | 8 | 684 | 47.2 | 69.0 | 84 % | 575 | 108 |
| MoE gate/up (K=2048 segmented) | 7 | 494 | 33.0 | 66.9 | 82 % | 403 | 91 |
| lm_head (K=2048, 150144 rows) | 1 | 2192 | 173.0 | 78.9 | 96 % | 2109 | 83 |
| router (bf16 32x2048, 32 warps = 4 of 16 WGs) | 7 | 50 | 0.9 | 18.5 | 23 % | 11 | 38 |
| dense gate+up / down (layer 0) | 1+1 | 108 | 7.1 | 65 | 80 % | 86 | 22 |
| **weight phases** | | **4566** | **315.4** | **69.1** | 84 % | 3847 | 719 |
| light phases + barriers | | 289 | | | | | |
| ...of which attention (8 layers) | | 127 | | | | | |

Ceilings: weights at roof = 3847 us -> 260 tok/s; with today's 289 us of light phases -> 242 tok/s.

## Reading -> next lever
- The loss is concentrated in the two NON-K=2048 weight phases: O-proj (K=4096) and MoE down (K=512/1024).
  Together they lose 378 us/token (7.8 %). The K=2048 fast path already runs at 82-96 % of roof.
- Activations are KBs against 315 MB of weights per token, so **relaxed mixed-precision boundaries buy ~0 bytes on
  this rig**. They would spend the ~1e-4 fp32_bf16kv headroom for no time. Not recommended.
- Next lever (bandwidth side, relaxed summation order only, ~1e-7-class error):
  1. **O-proj split-K onto the K=2048 segmented path**: two K=2048 halves (row stride 2048 B) into two partial
     outputs, summed in the existing residual phase (ws_x += o0 + o1). Target 56 -> ~67-69 GB/s: ~-125 us (~2.6 %).
  2. **MoE down as one K-concatenated virtual K=2048 gemv** ([act0*w0 | act1*w1 | act_sh] over three weight blocks
     with per-chunk base/stride). It replaces three short-row calls and folds the routing weights. Target 45 -> ~65 GB/s:
     ~-110 us (~2.3 %).
  3. Then router spread over all warps (~-35 us) and attention split across idle warps (~-90 us at short ctx).
