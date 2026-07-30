#!/usr/bin/env python3
"""Phase 0 of plan #20: dump RAW GGUF scale blocks plus the reference decode, for the C++ gate.

NEW FILE -- imports dump_real_weights.py rather than re-implementing anything, so the reference stays the one
function already regressed against marlin_gguf_ppu.cuh:gguf_q4k_scales.

Output (little endian):
    magic 'GSB0' | u32 ktype (4 = Q4_K, 3 = Q3_K, 2 = Q2_K) | u32 nblocks | u32 block_bytes | u32 groups
    then nblocks records of: block_bytes raw bytes, then groups*i32 sc, then groups*i32 mn (zeros if no min)

Usage: python3 dump_scale_blocks.py <file.gguf> [--n 4096] [--out scale_blocks_q4k.bin]
"""
import sys, os, struct, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_real_weights as R

ap = argparse.ArgumentParser()
ap.add_argument("gguf")
ap.add_argument("--nblocks", type=int, default=4096)
ap.add_argument("--out", default=None)
a = ap.parse_args()

g = R.Gguf(a.gguf)
# tensors[name] = (dims, ttype, off); GGML_Q4_K == 12
cand = [nm for nm, (dims, tt, off) in g.tensors.items() if tt == R.GGML_Q4_K]
if not cand:
    sys.exit("no GGML_TYPE_Q4_K tensor in this file")
name = cand[0]
dims, ttype, raw = g.raw(name)
print(f"tensor {name} dims={dims} type=Q4_K bytes={len(raw)}")

nsuper = len(raw) // 144
nb = min(a.nblocks, nsuper)
out = a.out or "scale_blocks_q4k.bin"
with open(out, "wb") as f:
    f.write(b"GSB0" + struct.pack("<IIII", 4, nb, 12, 8))
    for i in range(nb):
        rec = raw[i * 144:(i + 1) * 144]
        sc, mn = R.get_scale_min_k4(rec[4:16])
        f.write(rec[4:16])
        f.write(np.asarray(sc, dtype=np.int32).tobytes())
        f.write(np.asarray(mn, dtype=np.int32).tobytes())
print(f"wrote {nb} Q4_K scale blocks + reference sc/mn -> {out}")
