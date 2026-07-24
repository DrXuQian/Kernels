#!/usr/bin/env python3
"""Extract a REAL Q3_K slice from a GGUF and emit the BIT-PLANE concat test .bin (int2 low + int1 high).

Q3_K is natively 2-bit + 1-bit:  W = dl * (low2 + 4*high1 - 4),  dl = d*(sc6-32),  gs = 16.
The -4 center folds into the LOW plane's affine zero, so the two planes are two plain W*A16 GEMMs:

    int2 plane:  q = low2  (0..3)   scale = dl        zero = -4*dl
    int1 plane:  q = high1 (0..1)   scale = 4*dl      zero = 0
    D = D_low + D_high  ==  A @ dequant(Q3_K)      (exactly)

So this one .bin validates (a) int1 on REAL GGUF bits, (b) int2 on real bits, and (c) the A-concat, at once.

.bin layout (all little-endian; q planes are [K][N] to match the kernel's B orientation):
    <4 int32>  M, N, K, gs(=16)
    A     [M*K]        f16
    low2  [K*N]        u8   (0..3)
    high1 [K*N]        u8   (0..1)
    sc_lo [(K/gs)*N]   f16  = dl
    zr_lo [(K/gs)*N]   f16  = -4*dl
    sc_hi [(K/gs)*N]   f16  = 4*dl
    gold  [M*N]        f16  = A @ (sc_lo*low2 + zr_lo + sc_hi*high1)   [fp16-rounded scales, fp32 accumulate]

usage: python3 dump_q3_concat.py <file.gguf> [--tensor blk.0.attn_output.weight] [--m 128 --n 256 --k 256] [-o out.bin]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_real_weights import Gguf, unpack_q3k_expert, GGML_Q3_K   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensor", default="blk.0.attn_output.weight")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("-o", "--out", default="real_q3k_concat.bin")
    a = ap.parse_args()

    g = Gguf(a.gguf)
    dims, ttype, raw = g.raw(a.tensor)
    assert ttype == GGML_Q3_K, f"{a.tensor}: ggml type {ttype}, need Q3_K({GGML_Q3_K})"
    Kfull, Nfull = dims[0], dims[1]          # dims are fast-first: K is fast
    M, N, K, gs = a.m, a.n, a.k, 16
    assert K % 256 == 0 and K <= Kfull and N <= Nfull, f"need K%256==0, K<={Kfull}, N<={Nfull}"
    print(f"[q3concat] {a.tensor} full=[K={Kfull} N={Nfull}] slice M={M} N={N} K={K} gs={gs}")

    blk = np.frombuffer(raw, dtype=np.uint8).reshape(Nfull, Kfull // 256, 110)
    sl = np.ascontiguousarray(blk[:N, : K // 256, :]).tobytes()
    low2, high1, dl = unpack_q3k_expert(sl, N, K)        # [K,N] u8, [K,N] u8, [K//16,N] f32

    # native dequant (fp32 dl) vs the two-plane form with FP16-ROUNDED scales (what the kernel really sees)
    q3 = low2.astype(np.float32) + 4.0 * high1.astype(np.float32) - 4.0
    W_native = dl.repeat(gs, axis=0) * q3
    sc_lo = dl.astype(np.float16)
    zr_lo = (-4.0 * dl).astype(np.float16)
    sc_hi = (4.0 * dl).astype(np.float16)
    W_planes = (sc_lo.astype(np.float32).repeat(gs, axis=0) * low2.astype(np.float32)
                + zr_lo.astype(np.float32).repeat(gs, axis=0)
                + sc_hi.astype(np.float32).repeat(gs, axis=0) * high1.astype(np.float32))
    dw = np.abs(W_planes - W_native)
    print(f"[q3concat] two-plane vs native dequant: max|d|={dw.max():.3e} "
          f"(pure fp16 rounding of the 3 scale arrays)  W std={W_native.std():.4f}")

    rng = np.random.default_rng(1234)
    A = rng.integers(-3, 4, size=(M, K)).astype(np.float32) * 0.25   # fp16-exact
    gold = (A @ W_planes).astype(np.float16)                          # fp32 accumulate

    with open(a.out, "wb") as f:
        f.write(struct.pack("<4i", M, N, K, gs))
        f.write(np.ascontiguousarray(A, np.float16).tobytes())
        f.write(np.ascontiguousarray(low2, np.uint8).tobytes())
        f.write(np.ascontiguousarray(high1, np.uint8).tobytes())
        f.write(np.ascontiguousarray(sc_lo, np.float16).tobytes())
        f.write(np.ascontiguousarray(zr_lo, np.float16).tobytes())
        f.write(np.ascontiguousarray(sc_hi, np.float16).tobytes())
        f.write(np.ascontiguousarray(gold, np.float16).tobytes())

    gm = np.abs(gold.astype(np.float32))
    print(f"[q3concat] wrote {a.out} ({os.path.getsize(a.out)} B)")
    print(f"[q3concat] plane hist: low2={np.bincount(low2.ravel(), minlength=4).tolist()} "
          f"high1={np.bincount(high1.ravel(), minlength=2).tolist()}")
    print(f"[q3concat] golden: mean|.|={gm.mean():.4g} max|.|={gm.max():.4g} "
          f"zeros={(gm == 0).mean() * 100:.1f}%   (guards against a trivial all-zero MATCH)")


if __name__ == "__main__":
    main()
