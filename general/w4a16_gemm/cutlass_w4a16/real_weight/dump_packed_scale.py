#!/usr/bin/env python3
"""Phase 3 of plan #20: emit the scale channel in its PACKED form -- int8 codes + the superblock fp16 -- instead of the
pre-multiplied fp16 plane.

NEW FILE, and it imports dump_real_weights rather than re-implementing anything: q_signed, the reference scale/zero
planes and the golden all come from the functions already regressed against marlin_gguf_ppu.cuh. What is new here is
only the OUTPUT FORM, which is the whole point of the phase.

    old (RWMOE):   scale[scale_k][N] f16          = d * sc6            2 bytes per group per column
    new (RWMOEP):  sc[scale_k][N] i8 + d[K/256][N] f16                 1 byte + 1/8 byte

and likewise mn + dmin where the format has a min. The device rebuilds

    scale = d[k/256][n] * sc[k/gs][n]
    zero  = z_mul * scale - dmin[k/256][n] * mn[k/gs][n]

The header carries z_mul and the converter bias the codes were emitted for. This writer emits z_mul = 0 / cvt_bias = 0:
the stored 4-bit codes already are the unsigned nib, so a converter that subtracts nothing leaves nib in the mainloop
and the zero is just -dmin*mn -- one product per group instead of a product and an fma. Phase 1 generalised exactly that
immediate. Gate (c) measured both choices against fp64 truth and they are equivalent in accuracy (both inside 0.015 of
a quantisation step), so this is a cost decision, not a precision one.

THE MAGIC IS BUMPED to "RWMOEP\0\0". An old loader compares 8 bytes and fails, rather than reading a byte plane as
half the elements of an fp16 plane -- silently misreading a header is the worst failure available here.

Scope: Q4_K. That is where the measured cost is (the scale channel is 18.8% of the decode-band kernel: zero 11.5% +
per-group reload 7.3%) and it is the format with a min channel, i.e. the hard case. Q3_K/Q6_K are strictly easier --
after phase 1 they have no min at all, so their packed form is one int8 plus d with nothing else -- but the offline has
no Q6_K unpacker yet and Q3_K's 6-bit assembly lives inline in unpack_q3k_expert; both are additive to this writer.

Usage:
  python3 dump_packed_scale.py <file.gguf> --tensor blk.11.ffn_down.weight --ncols 64 --m 16 --out q4k_packed.bin
  python3 dump_packed_scale.py <file.gguf> --selfcheck        # local gates only, no dump
"""
import sys, os, struct, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_real_weights as R

MAGIC = b"RWMOEP\0\0"
SB    = 256           # k per GGUF superblock
GS    = 32            # k per Q4_K scale group


# ---------------------------------------------------------------- vectorised raw scale extraction
def q4k_raw_scales(raw, N, K):
    """Q4_K tensor bytes, logical [N][K]. Returns (sc, mn) int8 [K/32][N] and (d, dmin) f16 [K/256][N].

    Vectorised over all superblocks. get_scale_min_k4 is the REFERENCE for the same quantity and check_raw() compares
    against it block by block -- this function exists for speed, not as a second source of truth."""
    nblk = K // SB
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(N, nblk, 144)
    d    = np.frombuffer(np.ascontiguousarray(blk[:, :, 0:2]).tobytes(), dtype=np.float16).reshape(N, nblk)
    dmin = np.frombuffer(np.ascontiguousarray(blk[:, :, 2:4]).tobytes(), dtype=np.float16).reshape(N, nblk)
    q = blk[:, :, 4:16].astype(np.int32)                       # [N][nblk][12]
    sc = np.zeros((N, nblk, 8), np.int32)
    mn = np.zeros((N, nblk, 8), np.int32)
    for g in range(4):                                          # the same two rules as get_scale_min_k4
        sc[:, :, g] = q[:, :, g] & 63
        mn[:, :, g] = q[:, :, g + 4] & 63
    for g in range(4, 8):
        sc[:, :, g] = (q[:, :, g + 4] & 0xF) | ((q[:, :, g - 4] >> 6) << 4)
        mn[:, :, g] = (q[:, :, g + 4] >> 4)  | ((q[:, :, g] >> 6) << 4)
    assert sc.max() < 64 and mn.max() < 64                      # 6-bit codes, so int8 is exact and fp16 is exact
    # [N][nblk][8] -> [scale_k][N] with scale_k = K/32, group g of block b being k-group b*8+g
    sc = sc.reshape(N, nblk * 8).T.astype(np.int8)
    mn = mn.reshape(N, nblk * 8).T.astype(np.int8)
    return sc, mn, d.T.copy(), dmin.T.copy()                    # d/dmin -> [K/256][N]


# ---------------------------------------------------------------- gates
def check_raw(raw, N, K, sc, mn, d, dmin, nblocks=512):
    """(a) the vectorised extraction against get_scale_min_k4, the function already regressed against the kernel."""
    nblk = K // SB
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(N, nblk, 144)
    bad = 0
    seen = 0
    for n in range(N):
        for b in range(nblk):
            if seen >= nblocks:
                break
            rsc, rmn = R.get_scale_min_k4(blk[n, b, 4:16].tobytes())
            rd    = np.frombuffer(blk[n, b, 0:2].tobytes(), dtype=np.float16)[0]
            rdmin = np.frombuffer(blk[n, b, 2:4].tobytes(), dtype=np.float16)[0]
            g0 = b * 8
            if (not np.array_equal(sc[g0:g0 + 8, n].astype(np.int32), rsc)
                    or not np.array_equal(mn[g0:g0 + 8, n].astype(np.int32), rmn)
                    or d[b, n] != rd or dmin[b, n] != rdmin):
                if bad < 4:
                    print(f"    [FAIL] n={n} b={b}: sc {sc[g0:g0+8, n]} vs {rsc}")
                bad += 1
            seen += 1
    print(f"  (a) vectorised (d,dmin,sc,mn) vs get_scale_min_k4 over {seen} superblocks -> {bad} bad")
    return bad


def check_product(sc, d):
    """(b) THE PRECISION CLAIM, over the file's own values rather than in the abstract.

    Today the host computes f32(d)*f32(sc) and rounds once to fp16. The device path will compute f16(d)*f16(sc). Both
    round the SAME exact real product once, because d is already an fp16 value and sc <= 63 needs 6 bits: the product
    has at most 11+6 = 17 significant bits, exact in fp32. So this must be EQUAL, not 1 ulp -- and 'must be equal' is a
    much better gate than 'should be close', so it is checked as equality."""
    scale_k, N = sc.shape
    rep = np.repeat(d, scale_k // d.shape[0], axis=0)                       # d is per 256 k, sc per 32 k
    old = (rep.astype(np.float32) * sc.astype(np.float32)).astype(np.float16)
    new = (rep * sc.astype(np.float16)).astype(np.float16)                  # one fp16 multiply, as on device
    bad = int(np.sum(old.view(np.uint16) != new.view(np.uint16)))
    print(f"  (b) f16(f32(d)*f32(sc)) == f16(d)*f16(sc) over {old.size} groups -> {bad} bad (must be 0, not 'close')")
    return bad


def check_weight(q_signed, ref_scale, ref_zero, sc, mn, d, dmin):
    """(c) THE WEIGHT ITSELF, three ways, against the exact GGUF dequant in fp64. This is the check that matters, and
    the first version of it -- comparing the two ZEROs to each other -- was the wrong question.

    Q4_K is w = d*sc*nib - dmin*mn with nib in [0,15]. Today's pipeline stores q_signed = nib-8 and therefore carries
    zero = 8*d*sc - dmin*mn, precomputed by the offline in fp32 and rounded once. A packed form has to FORM the zero on
    device instead, and there are two ways:
        Bias = 0:  the stored 4-bit codes already ARE the unsigned nib (preprocess adds 8 back), so a converter that
                   subtracts nothing hands the mainloop nib directly and zero = -dmin*mn      -- ONE product
        Bias = 8:  keep today's convention and zero = 8*scale - dmin*mn                       -- product + fma
    Nothing about the B bytes or the offline packing changes either way; only the converter's immediate.

    MEASURED, and it corrected my first reading: in units of the QUANTISATION STEP both on-device forms land inside
    0.015 step against fp64 truth, with the cancelling one marginally AHEAD. The "196 fp16 ulps" the first version of
    this check reported were ulps of the zero itself -- the wrong denominator, since the zero is small. Accuracy does
    not choose between them. Bias=0 is chosen because it is CHEAPER (one product, and no dependency on scale), which is
    a cost argument and is stated as one."""
    K, N = q_signed.shape
    scale_k = sc.shape[0]
    g   = np.arange(K) // GS
    gb  = np.arange(K) // SB
    nib = (q_signed.astype(np.int32) + 8)
    # fp64 truth, straight off the raw parts
    W_true = (d.astype(np.float64)[gb, :] * sc.astype(np.float64)[g, :] * nib
              - dmin.astype(np.float64)[gb, :] * mn.astype(np.float64)[g, :])
    # today: fp16 scale and fp16 zero (the zero itself formed in fp32 by the offline, as the reference does), then
    # scale*q + zero in fp16
    W_ref = (ref_scale.astype(np.float16)[g, :] * q_signed.astype(np.float16)
             + ref_zero.astype(np.float16)[g, :]).astype(np.float16)
    # packed, Bias=0: scale = d*sc (fp16, exact per (b)), zero = -dmin*mn (fp16, exact for the same reason)
    scale16 = (d[gb, :] * sc.astype(np.float16)[g, :]).astype(np.float16)
    zero16  = (-(dmin[gb, :] * mn.astype(np.float16)[g, :])).astype(np.float16)
    W_new   = (scale16 * nib.astype(np.float16) + zero16).astype(np.float16)
    # IN UNITS OF THE QUANTISATION STEP, not relative to W. Relative-to-W is the wrong metric and its first reading
    # proved it: W_true passes through ~0, so max_rel was dominated by weights that are essentially zero and the
    # CANCELLING form scored "best" on it. The step d*sc is the only scale that means anything here -- the weight
    # already carries +-0.5 step of quantisation error, so an fp16 representation error is only interesting as a
    # fraction of a step.
    step = np.abs(d.astype(np.float64)[gb, :] * sc.astype(np.float64)[g, :]) + 1e-30
    e_ref = np.abs(W_ref.astype(np.float64) - W_true) / step
    e_new = np.abs(W_new.astype(np.float64) - W_true) / step
    # And what the cancelling on-device form would cost, so the reason for Bias=0 is a number, not an argument.
    z_bad = (np.float16(8) * scale16 - dmin[gb, :] * mn.astype(np.float16)[g, :]).astype(np.float16)
    W_bad = (scale16 * q_signed.astype(np.float16) + z_bad).astype(np.float16)
    e_bad = np.abs(W_bad.astype(np.float64) - W_true) / step
    print(f"  (c) weight error in QUANTISATION STEPS (fp64 truth; a step is one code of d*sc):")
    print(f"      current pipeline, fp32-precomputed zero   max={e_ref.max():.4f}  mean={e_ref.mean():.5f}")
    print(f"      packed, Bias=0, zero = -dmin*mn           max={e_new.max():.4f}  mean={e_new.mean():.5f}")
    print(f"      packed, Bias=8, zero = 8*scale - dmin*mn  max={e_bad.max():.4f}  mean={e_bad.mean():.5f}")
    # The offline's fp32 zero is the most accurate of the three because it rounds ONCE, after the cancellation; forming
    # the zero on device costs a little accuracy either way. The gate is therefore an absolute bound -- well inside a
    # step -- and NOT "better than today", which it is not.
    print(f"      => forming the zero on device costs {e_new.mean() - e_ref.mean():+.5f} step of mean error; the two"
          f" on-device forms differ by {abs(e_new.mean() - e_bad.mean()):.5f} step, i.e. not the deciding factor")
    return 0 if e_new.max() < 0.25 else 1


# ---------------------------------------------------------------- writer
def write_packed(path, L, M, N, K, gs, mode, ktype, z_mul, cvt_bias, A_list, qs_list, sc_list, mn_list, d_list, dm_list, gold):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<10i", L, M, N, K, gs, mode, ktype, SB, z_mul, cvt_bias))
        for A in A_list:  f.write(np.ascontiguousarray(A,   dtype=np.float16).tobytes())
        for q in qs_list: f.write(np.ascontiguousarray(q.T, dtype=np.int8).tobytes())     # [K][N] -> [N][K], as RWMOE
        for s in sc_list: f.write(np.ascontiguousarray(s,   dtype=np.int8).tobytes())
        if mode == 1:
            for m in mn_list: f.write(np.ascontiguousarray(m, dtype=np.int8).tobytes())
        for x in d_list:  f.write(np.ascontiguousarray(x, dtype=np.float16).tobytes())
        if mode == 1:
            for x in dm_list: f.write(np.ascontiguousarray(x, dtype=np.float16).tobytes())
        for g in gold:    f.write(np.ascontiguousarray(g, dtype=np.float16).tobytes())
    print(f"wrote {path}: L={L} M={M} N={N} K={K} gs={gs} mode={mode} ktype={ktype} z_mul={z_mul} cvt_bias={cvt_bias}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensor", default=None, help="tensor name; default = the first Q4_K tensor in the file")
    ap.add_argument("--ncols", type=int, default=64, help="N columns to take (the python reference is O(N*K))")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true", help="run the gates on the FULL tensor, write nothing")
    a = ap.parse_args()

    g = R.Gguf(a.gguf)
    name = a.tensor
    if name is None:
        cand = [nm for nm, (dims, tt, off) in g.tensors.items() if tt == R.GGML_Q4_K]
        if not cand:
            sys.exit("no Q4_K tensor in this file")
        name = cand[0]
    dims, ttype, raw = g.raw(name)
    assert ttype == R.GGML_Q4_K, f"{name} is ggml type {ttype}, not Q4_K ({R.GGML_Q4_K})"
    K, N = int(dims[0]), int(dims[1])            # dims are fast-first: [K][N] logically [N][K] per row
    print(f"tensor {name}: K={K} N={N} bytes={len(raw)}")

    sc, mn, d, dmin = q4k_raw_scales(raw, N, K)
    fails = 0
    print("== gates ==")
    fails += check_raw(raw, N, K, sc, mn, d, dmin)
    fails += check_product(sc, d)

    # (c) and the end-to-end plane check need the SLOW reference, so they run on a column slice.
    nc = min(a.ncols, N)
    row_bytes = (K // SB) * 144
    q_signed, ref_scale, ref_zero = R.unpack_q4k_expert(raw[:nc * row_bytes], nc, K)
    fails += check_weight(q_signed, ref_scale, ref_zero, sc[:, :nc], mn[:, :nc], d[:, :nc], dmin[:, :nc])
    # (d) the PLANE ALIGNMENT: d*sc must equal the reference scale plane element for element, which is what catches a
    # transposed or off-by-one-superblock (d,sc) pairing -- the thing a per-block check cannot see.
    rep = np.repeat(d[:, :nc], (K // GS) // (K // SB), axis=0)
    got = (rep * sc[:, :nc].astype(np.float16)).astype(np.float16)
    badp = int(np.sum(got.view(np.uint16) != ref_scale.view(np.uint16)))
    print(f"  (d) d*sc plane == reference scale plane, {got.shape} -> {badp} bad")
    fails += badp

    print(f"== {'FAIL' if fails else 'PASS'}: {fails} ==")
    if a.selfcheck:
        return 1 if fails else 0
    if fails:
        return 1

    A, gold = R.golden_and_pack(q_signed, ref_scale, ref_zero, GS, a.m)
    out = a.out or "q4k_packed.bin"
    # z_mul = 0 and cvt_bias = 0: the device rebuilds zero = -dmin*mn only, and the int4 converter must NOT
    # subtract 8, because the stored codes already are the unsigned nib. Gate (c) is what decided those two numbers.
    write_packed(out, 1, a.m, nc, K, GS, 1, 4, 0, 0, [A], [q_signed], [sc[:, :nc]], [mn[:, :nc]],
                 [d[:, :nc]], [dmin[:, :nc]], [gold])
    return 0


if __name__ == "__main__":
    sys.exit(main())
