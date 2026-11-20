#!/usr/bin/env bash
# Compile-only resource report per gemv function and kernel, for each .hip given. No GPU use.
# Usage: res.sh a.hip [b.hip ...] [-- extra hipcc flags]
source ~/opt/rocm-env.sh >/dev/null 2>&1
files=(); flags=()
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; flags=("$@"); break; }; files+=("$1"); shift; done
for f in "${files[@]}"; do
  d=$(mktemp -d); cp "$f" $d/k.hip
  (cd $d && hipcc -O2 -shared -fPIC --offload-arch=gfx1150 --save-temps "${flags[@]}" k.hip -o k.so > build.log 2>&1) || { echo "$f BUILD FAILED"; grep -m3 error $d/build.log; rm -rf $d; continue; }
  S=$(ls $d/*gfx1150.s)
  printf "== %s\n" "$f"
  grep -E "^\s+\.set\s+\.L\S+\.(num_vgpr|private_seg_size),\s+[0-9]+\s*$" $S | \
    sed -E 's/^\s+\.set\s+\.L//; s/,//' | \
    awk '{n=$1; v=$2; k=n; sub(/\.(num_vgpr|private_seg_size)$/, "", k); t=(n ~ /num_vgpr$/) ? "v" : "s"; r[k","t]=v; ks[k]=1}
         END{for (k in ks) if (k ~ /gemv_op|^mid_generate$/) { nm=k; sub(/ELb0EEvPKh.*/, "", nm); printf "  %-36s vgpr=%-4s scratch=%s\n", nm, r[k",v"], r[k",s"] }}' | sort
  grep -E "^; (NumVgprs|ScratchSize):" $S | paste - - | sort | uniq -c | sort -rn | head -3 | sed 's/^/  kernels: /'
  rm -rf $d
done
