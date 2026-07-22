#!/usr/bin/env python3
"""GGUF K-quant / legacy-quant unpackers -> unified intermediate for the actlize mixed-input collective.

Each unpack_<fmt>(raw_bytes, N, K) returns, for one expert weight logically [N(out)][K(in)] (K/QK blocks per row):
    q_signed [K][N] int8   -- the quantized value in the collective's convention (w = scale*q_signed + zero)
    scale    [K//gs][N] float16
    zero     [K//gs][N] float16  or None (symmetric formats: Q3_K, Q6_K, Q8_0)
    gs       int          -- group size along K
    slot_bits int         -- 4 (q_signed fits int4 [-8,7]) or 8 (int8), i.e. which collective (W4A16 / W8A16)

Element order follows llama.cpp's dequantize_row_qX_K exactly (the y++ order == the k index within a row), so a
real-inference cross-check vs llama.cpp is meaningful. Ported from ggml-quants.c; verified here by a synthetic
round-trip vs an independent element-by-element reference (python3 gguf_dequant.py).

Convention notes (matches the collective's +bias-and-convert, same as Q4_K in dump_real_weights.py):
  * The additive centering (q - 2^(b-1)) for symmetric K-quants is FOLDED into `zero` where a min exists, else the
    value is stored pre-centered in q_signed (fits the signed slot). Q8_0/Q3_K/Q6_K are symmetric (zero=None).
  * gs=16 formats (Q2_K/Q3_K/Q6_K) need the collective's FINE per-mma-atom scale path (Scale_TileK<=TK/16).
"""
import numpy as np

BLOCK_BYTES = {"q8_0": 34, "q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210}
GGML_TYPE = {2: "q4_0", 3: "q4_1", 8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k"}


def _half(b2):  # bytes[...,2] uint8 -> float32 (view last axis pair as float16)
    return b2.copy().view(np.float16).astype(np.float32)


def _place_KN(per_blk, N, K, inner):
    """per_blk: [N, nblk, inner] -> [K][N] with k = blk*inner + local (row-major over blocks then local)."""
    nblk = per_blk.shape[1]
    return per_blk.reshape(N, nblk * inner).T.copy()


def get_scale_min_k4(scbytes12):  # [.,.,12] uint8 -> (sc[.,.,8], mn[.,.,8]) int32  (llama.cpp get_scale_min_k4)
    q = scbytes12.astype(np.int32)
    sc = np.zeros(q.shape[:-1] + (8,), np.int32)
    mn = np.zeros(q.shape[:-1] + (8,), np.int32)
    for g in range(4):
        sc[..., g] = q[..., g] & 63
        mn[..., g] = q[..., g + 4] & 63
    for g in range(4, 8):
        sc[..., g] = (q[..., g + 4] & 0xF) | ((q[..., g - 4] >> 6) << 4)
        mn[..., g] = (q[..., g + 4] >> 4) | ((q[..., g - 0] >> 6) << 4)
    return sc, mn


# ----------------------------------------------------------------------------- Q8_0 (32/block, gs=32, sym)
def unpack_q8_0(raw, N, K):
    gs = 32; nblk = K // 32
    b = np.frombuffer(raw, np.uint8).reshape(N, nblk, 34)
    d = _half(b[:, :, 0:2])[:, :, 0]                       # [N,nblk]
    qs = b[:, :, 2:34].view(np.int8).astype(np.int8)       # [N,nblk,32]
    q_signed = _place_KN(qs.astype(np.int8), N, K, 32)     # [K][N]
    scale = (d[:, :, None] * np.ones((1, 1, 1), np.float32)).astype(np.float16)  # [N,nblk,1]
    scale = _place_KN(np.broadcast_to(d[:, :, None], (N, nblk, 1)).astype(np.float16), N, K // gs, 1)  # [K//gs][N]
    return q_signed, scale, None, gs, 8


# ----------------------------------------------------------------------------- Q6_K (256/block, gs=16, sym)
def unpack_q6k(raw, N, K):
    gs = 16; nblk = K // 256
    b = np.frombuffer(raw, np.uint8).reshape(N, nblk, 210)
    ql = b[:, :, 0:128].astype(np.int32)                   # [N,nblk,128]
    qh = b[:, :, 128:192].astype(np.int32)                 # [N,nblk,64]
    sc = b[:, :, 192:208].view(np.int8).astype(np.float32) # [N,nblk,16] int8 scales
    d = _half(b[:, :, 208:210])[:, :, 0]                   # [N,nblk]
    q6 = np.zeros((N, nblk, 256), np.int32)
    l = np.arange(32)
    for g in range(2):
        qlg = ql[:, :, g * 64:(g + 1) * 64]
        qhg = qh[:, :, g * 32:(g + 1) * 32]
        q6[:, :, g * 128 + 0 * 32 + l] = (qlg[:, :, l] & 0xF) | ((qhg[:, :, l] & 3) << 4)
        q6[:, :, g * 128 + 1 * 32 + l] = (qlg[:, :, l + 32] & 0xF) | (((qhg[:, :, l] >> 2) & 3) << 4)
        q6[:, :, g * 128 + 2 * 32 + l] = (qlg[:, :, l] >> 4) | (((qhg[:, :, l] >> 4) & 3) << 4)
        q6[:, :, g * 128 + 3 * 32 + l] = (qlg[:, :, l + 32] >> 4) | (((qhg[:, :, l] >> 6) & 3) << 4)
    q_signed = _place_KN((q6 - 32).astype(np.int8), N, K, 256)          # [K][N]
    scale = _place_KN((d[:, :, None] * sc).astype(np.float16), N, K // gs, 16)  # [K//16][N]
    return q_signed, scale, None, gs, 8


# ----------------------------------------------------------------------------- Q5_K (256/block, gs=32, affine)
def unpack_q5k(raw, N, K):
    gs = 32; nblk = K // 256
    b = np.frombuffer(raw, np.uint8).reshape(N, nblk, 176)
    d = _half(b[:, :, 0:2])[:, :, 0]                       # [N,nblk]
    dmin = _half(b[:, :, 2:4])[:, :, 0]
    sc, mn = get_scale_min_k4(b[:, :, 4:16])               # [N,nblk,8]
    qh = b[:, :, 16:48].astype(np.int32)                   # [N,nblk,32]
    qs = b[:, :, 48:176].astype(np.int32)                  # [N,nblk,128]
    q5 = np.zeros((N, nblk, 256), np.int32)
    l = np.arange(32)
    for jj in range(4):
        qsj = qs[:, :, jj * 32:(jj + 1) * 32]              # ql advances +32 per iter
        u1 = 1 << (2 * jj); u2 = 2 << (2 * jj)
        q5[:, :, jj * 64 + 0 * 32 + l] = (qsj[:, :, l] & 0xF) + np.where(qh[:, :, l] & u1, 16, 0)
        q5[:, :, jj * 64 + 1 * 32 + l] = (qsj[:, :, l] >> 4) + np.where(qh[:, :, l] & u2, 16, 0)
    q_signed = _place_KN((q5 - 16).astype(np.int8), N, K, 256)          # [-16,15]
    sv = (d[:, :, None] * sc)                              # [N,nblk,8]
    zero = (-dmin[:, :, None] * mn + 16.0 * sv)            # fold the -16 centering
    scale = _place_KN(sv.astype(np.float16), N, K // gs, 8)
    zero = _place_KN(zero.astype(np.float16), N, K // gs, 8)
    return q_signed, scale, zero, gs, 8


# ----------------------------------------------------------------------------- Q2_K (256/block, gs=16, affine)
def unpack_q2k(raw, N, K):
    gs = 16; nblk = K // 256
    b = np.frombuffer(raw, np.uint8).reshape(N, nblk, 84)
    scales = b[:, :, 0:16].astype(np.int32)                # [N,nblk,16]
    qs = b[:, :, 16:80].astype(np.int32)                   # [N,nblk,64]
    d = _half(b[:, :, 80:82])[:, :, 0]
    dmin = _half(b[:, :, 82:84])[:, :, 0]
    q2 = np.zeros((N, nblk, 256), np.int32)
    l = np.arange(16)
    for g in range(2):
        qg = qs[:, :, g * 32:(g + 1) * 32]                 # q advances +32 per n-group
        for j in range(4):
            sh = 2 * j
            q2[:, :, g * 128 + j * 32 + 0 * 16 + l] = (qg[:, :, l] >> sh) & 3
            q2[:, :, g * 128 + j * 32 + 1 * 16 + l] = (qg[:, :, l + 16] >> sh) & 3
    q_signed = _place_KN(q2.astype(np.int8), N, K, 256)                 # [0,3]
    scl = (d[:, :, None] * (scales & 0xF)).astype(np.float32)           # [N,nblk,16]
    zero = (-dmin[:, :, None] * (scales >> 4)).astype(np.float32)
    scale = _place_KN(scl.astype(np.float16), N, K // gs, 16)
    zero = _place_KN(zero.astype(np.float16), N, K // gs, 16)
    return q_signed, scale, zero, gs, 4


# ----------------------------------------------------------------------------- Q3_K (256/block, gs=16, sym)
def unpack_q3k(raw, N, K):
    gs = 16; nblk = K // 256
    b = np.frombuffer(raw, np.uint8).reshape(N, nblk, 110)
    hmask = b[:, :, 0:32].astype(np.int32)                 # [N,nblk,32]
    qs = b[:, :, 32:96].astype(np.int32)                   # [N,nblk,64]
    scbytes = b[:, :, 96:108].astype(np.uint32)            # 12 bytes -> 16 six-bit scales (aux scheme)
    d = _half(b[:, :, 108:110])[:, :, 0]
    # unpack 16 six-bit scales via the ggml aux manipulation (per (N,nblk))
    a0 = scbytes[:, :, 0] | (scbytes[:, :, 1] << 8) | (scbytes[:, :, 2] << 16) | (scbytes[:, :, 3] << 24)
    a1 = scbytes[:, :, 4] | (scbytes[:, :, 5] << 8) | (scbytes[:, :, 6] << 16) | (scbytes[:, :, 7] << 24)
    tmp = scbytes[:, :, 8] | (scbytes[:, :, 9] << 8) | (scbytes[:, :, 10] << 16) | (scbytes[:, :, 11] << 24)
    km1 = np.uint32(0x03030303); km2 = np.uint32(0x0f0f0f0f)
    A2 = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)
    A3 = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)
    A0 = (a0 & km2) | (((tmp >> 0) & km1) << 4)
    A1 = (a1 & km2) | (((tmp >> 2) & km1) << 4)
    sc16 = np.zeros((N, nblk, 16), np.int32)               # int8 lanes of A0..A3
    for i, A in enumerate((A0, A1, A2, A3)):
        for byte in range(4):
            sc16[:, :, i * 4 + byte] = (A >> (8 * byte)) & 0xFF
    sc16 = sc16 - 32                                        # 6-bit biased -> signed
    q3 = np.zeros((N, nblk, 256), np.int32)
    l = np.arange(16)
    for g in range(2):
        qg = qs[:, :, g * 32:(g + 1) * 32]                 # q advances +32 per n-group
        for j in range(4):
            sh = 2 * j; m = 1 << (g * 4 + j)
            q2a = (qg[:, :, l] >> sh) & 3
            q2b = (qg[:, :, l + 16] >> sh) & 3
            ha = np.where(hmask[:, :, l] & m, 0, 4)
            hb = np.where(hmask[:, :, l + 16] & m, 0, 4)
            q3[:, :, g * 128 + j * 32 + 0 * 16 + l] = q2a - ha
            q3[:, :, g * 128 + j * 32 + 1 * 16 + l] = q2b - hb
    q_signed = _place_KN(q3.astype(np.int8), N, K, 256)                 # [-4,3]
    scale = _place_KN((d[:, :, None] * sc16).astype(np.float16), N, K // gs, 16)
    return q_signed, scale, None, gs, 4


UNPACK = {"q8_0": unpack_q8_0, "q6_k": unpack_q6k, "q5_k": unpack_q5k, "q2_k": unpack_q2k, "q3_k": unpack_q3k}


# ================================================================= self-test: synthetic block round-trip
def _ref_dequant(fmt, raw, N, K):
    """Independent element-by-element dequant (straight llama.cpp transcription) -> W[K][N] float32, for tests."""
    nblk = K // (32 if fmt == "q8_0" else 256)
    W = np.zeros((K, N), np.float32)
    bb = np.frombuffer(raw, np.uint8).reshape(N, nblk, BLOCK_BYTES[fmt])
    for n in range(N):
        for blk in range(nblk):
            rec = bb[n, blk]
            y = []
            if fmt == "q8_0":
                d = rec[0:2].copy().view(np.float16)[0].astype(np.float32)
                qs = rec[2:34].view(np.int8)
                y = [d * int(qs[i]) for i in range(32)]
            elif fmt == "q6_k":
                ql = rec[0:128].astype(np.int32); qh = rec[128:192].astype(np.int32)
                sc = rec[192:208].view(np.int8).astype(np.int32); d = rec[208:210].copy().view(np.float16)[0].astype(np.float32)
                yy = [0.0] * 256
                for g in range(2):
                    for ll in range(32):
                        isc = ll // 16
                        q1 = ((ql[g*64+ll] & 0xF) | ((qh[g*32+ll] & 3) << 4)) - 32
                        q2 = ((ql[g*64+ll+32] & 0xF) | (((qh[g*32+ll] >> 2) & 3) << 4)) - 32
                        q3 = ((ql[g*64+ll] >> 4) | (((qh[g*32+ll] >> 4) & 3) << 4)) - 32
                        q4 = ((ql[g*64+ll+32] >> 4) | (((qh[g*32+ll] >> 6) & 3) << 4)) - 32
                        yy[g*128+ll]    = d * sc[g*8+isc+0] * q1
                        yy[g*128+ll+32] = d * sc[g*8+isc+2] * q2
                        yy[g*128+ll+64] = d * sc[g*8+isc+4] * q3
                        yy[g*128+ll+96] = d * sc[g*8+isc+6] * q4
                y = yy
            elif fmt == "q5_k":
                d = rec[0:2].copy().view(np.float16)[0].astype(np.float32); dmin = rec[2:4].copy().view(np.float16)[0].astype(np.float32)
                sc, mn = get_scale_min_k4(rec[4:16][None, None, :]); sc = sc[0, 0]; mn = mn[0, 0]
                qh = rec[16:48].astype(np.int32); qs = rec[48:176].astype(np.int32)
                yy = [0.0] * 256; idx = 0
                for jj in range(4):
                    u1 = 1 << (2*jj); u2 = 2 << (2*jj)
                    for ll in range(32):
                        yy[idx] = d*sc[2*jj]*((qs[jj*32+ll] & 0xF) + (16 if qh[ll] & u1 else 0)) - dmin*mn[2*jj]; idx += 1
                    for ll in range(32):
                        yy[idx] = d*sc[2*jj+1]*((qs[jj*32+ll] >> 4) + (16 if qh[ll] & u2 else 0)) - dmin*mn[2*jj+1]; idx += 1
                y = yy
            elif fmt == "q2_k":
                scales = rec[0:16].astype(np.int32); qs = rec[16:80].astype(np.int32)
                d = rec[80:82].copy().view(np.float16)[0].astype(np.float32); dmin = rec[82:84].copy().view(np.float16)[0].astype(np.float32)
                yy = [0.0] * 256; idx = 0; iss = 0
                for g in range(2):
                    for j in range(4):
                        sh = 2*j
                        s = scales[iss]; iss += 1
                        for ll in range(16): yy[idx] = d*(s & 0xF)*((qs[g*32+ll] >> sh) & 3) - dmin*(s >> 4); idx += 1
                        s = scales[iss]; iss += 1
                        for ll in range(16): yy[idx] = d*(s & 0xF)*((qs[g*32+ll+16] >> sh) & 3) - dmin*(s >> 4); idx += 1
                y = yy
            elif fmt == "q3_k":
                hmask = rec[0:32].astype(np.int32); qs = rec[32:96].astype(np.int32)
                sb = rec[96:108].astype(np.uint32); d = rec[108:110].copy().view(np.float16)[0].astype(np.float32)
                a0 = sb[0]|(sb[1]<<8)|(sb[2]<<16)|(sb[3]<<24); a1 = sb[4]|(sb[5]<<8)|(sb[6]<<16)|(sb[7]<<24)
                tmp = sb[8]|(sb[9]<<8)|(sb[10]<<16)|(sb[11]<<24)
                km1 = np.uint32(0x03030303); km2 = np.uint32(0x0f0f0f0f)
                A = [np.uint32((a0 & km2) | (((tmp>>0)&km1)<<4)), np.uint32((a1 & km2) | (((tmp>>2)&km1)<<4)),
                     np.uint32(((a0>>4)&km2) | (((tmp>>4)&km1)<<4)), np.uint32(((a1>>4)&km2) | (((tmp>>6)&km1)<<4))]
                sc16 = [int((A[i] >> (8*byte)) & 0xFF) - 32 for i in range(4) for byte in range(4)]
                yy = [0.0]*256; idx = 0
                for g in range(2):
                    for j in range(4):
                        sh = 2*j; m = 1 << (g*4+j)
                        for ll in range(16):
                            yy[idx] = d*sc16[g*8+j*2+0]*(((qs[g*32+ll]>>sh)&3) - (0 if hmask[ll] & m else 4)); idx += 1
                        for ll in range(16):
                            yy[idx] = d*sc16[g*8+j*2+1]*(((qs[g*32+ll+16]>>sh)&3) - (0 if hmask[ll+16] & m else 4)); idx += 1
                y = yy
            for k in range(len(y)):
                W[blk * len(y) + k, n] = y[k]
    return W


# half-field byte offsets to overwrite with SANE d/dmin (random bytes -> absurd fp16 -> nan; real d ~ 0.01)
_DFIELDS = {"q8_0": [(0, 0.01)], "q6_k": [(208, 0.01)], "q5_k": [(0, 0.02), (2, 0.006)],
            "q2_k": [(80, 0.02), (82, 0.006)], "q3_k": [(108, 0.02)]}

def _selftest():
    rng = np.random.default_rng(3)
    for fmt, unpack in UNPACK.items():
        QK = 32 if fmt == "q8_0" else 256
        N, K = 2, QK * 2
        nblk = K // QK
        blk_bytes = BLOCK_BYTES[fmt]
        arr = rng.integers(0, 256, (N, nblk, blk_bytes), dtype=np.uint8)
        for off, val in _DFIELDS[fmt]:            # set sane d/dmin
            hb = np.frombuffer(np.float16(val).tobytes(), np.uint8)
            arr[:, :, off] = hb[0]; arr[:, :, off + 1] = hb[1]
        raw = arr.tobytes()
        q, sc, zr, gs, bits = unpack(raw, N, K)
        Wu = sc.astype(np.float32)[(np.arange(K) // gs), :] * q.astype(np.float32)
        if zr is not None:
            Wu = Wu + zr.astype(np.float32)[(np.arange(K) // gs), :]
        Wr = _ref_dequant(fmt, raw, N, K)
        err = np.abs(Wu - Wr).max()
        rng2 = np.abs(Wr).max() + 1e-6
        ok = err < 1e-2 * rng2 or err < 1e-2
        print(f"  {fmt:5s} slot=int{bits} gs={gs}  q_range=[{int(q.min())},{int(q.max())}]  "
              f"max|W_unpack-W_ref|={err:.4g} (Wref max {np.abs(Wr).max():.4g})  {'OK' if ok else 'BAD'}")


if __name__ == "__main__":
    print("gguf_dequant self-test (synthetic blocks, vectorized-vs-loop reference):")
    _selftest()
