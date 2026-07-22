#!/usr/bin/env python3
"""Extract REAL MoE expert weights from a Qwen3.5 GPTQ-Int4 (safetensors) or Q4_K_M (GGUF) checkpoint,
map them into the actlize collective's convention, compute an algorithm-independent fp16 golden, and dump
everything to a .bin that test_moe_grouped_real.cu loads.

Pure numpy (no torch / no safetensors / no gguf libs needed — safetensors + GGUF parsed by hand).

Convention (matches the collective; verified against marlin_gguf_ppu.cuh + the GPTQ config):
  Per (out=n, in=k):  dequant  W[k,n] = scale_g * q_signed + zero_g   with q_signed in [-8,7], g = k // gs.
  * GPTQ (sym):  q_signed = q_u - 8 ;  scale = scales[g,n] ;  zero = 0.        (bits=4, gs=128, desc_act=false)
  * Q4_K:        q_signed = nib(n,k) - 8 ;  scale = d*sc6[b] ;  zero = -dmin*mn6[b] + 8*scale ;  gs=32.
  The C++ driver feeds q_signed (packed) through preprocess_weights_for_mixed_gemm<false,256>, and scale/zero
  as fp16 [scale_k][N] row-major (sk outer, n inner) -- which is exactly GPTQ's scales layout.

Dump format (little-endian), consumed by test_moe_grouped_real.cu:
  char[8]  "RWMOE\0\0\0"
  int32    L, M, N, K, gs, quant_mode          (quant_mode: 0=scale-only, 1=scale+zero)
  float16  A            [L][M][K]               (random activations; real weights are the point)
  int8     q_signed     [L][K][N]               (in [-8,7])   -- collective B logical [K,N] row-major
  float16  scale        [L][scale_k][N]         (scale_k = K/gs)
  float16  zero         [L][scale_k][N]         (present only if quant_mode==1)
  float16  golden       [L][M][N]               (= A @ dequant(W), fp32 accumulate -> fp16)

Usage:
  python3 dump_real_weights.py gptq <dir/ow7_224_ca> --layer 0 --experts 0-7 --proj gate --m 128 --out g.bin
  python3 dump_real_weights.py gguf <file.gguf>       --layer 0 --experts 0-7 --proj gate --m 128 --out q.bin
  python3 dump_real_weights.py gguf <file.gguf> --selfcheck        # local sanity of the Q4_K unpack, no dump
"""
import sys, os, json, struct, argparse, glob
import numpy as np


# =========================================================== safetensors (manual, mmap)
def st_open(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
    data_start = 8 + n
    return f, hdr, data_start

_ST_DT = {"I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
          "F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I64": np.int64}

def st_get(f, hdr, data_start, name):
    m = hdr[name]
    dt = _ST_DT[m["dtype"]]
    b0, b1 = m["data_offsets"]
    f.seek(data_start + b0)
    buf = f.read(b1 - b0)
    arr = np.frombuffer(buf, dtype=dt).reshape(m["shape"])
    return arr


# =========================================================== GGUF (manual parser + Q4_K unpack)
_GT_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_GT_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGML_Q4_K = 12

class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        assert f.read(4) == b"GGUF"
        self.version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            vt = struct.unpack("<I", f.read(4))[0]
            self.kv[k] = self._val(vt)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            self.tensors[name] = (dims, ttype, off)
        align = self.kv.get("general.alignment", 32)
        pos = f.tell()
        self.data_start = (pos + align - 1) // align * align

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, vt):
        if vt == 8:
            return self._str()
        if vt == 9:
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            if et == 8:
                return [self._str() for _ in range(n)]
            return [struct.unpack(_GT_FMT[et], self.f.read(_GT_SZ[et]))[0] for _ in range(n)]
        return struct.unpack(_GT_FMT[vt], self.f.read(_GT_SZ[vt]))[0]

    def raw(self, name):
        dims, ttype, off = self.tensors[name]
        # dims are stored fastest-first (dims[0] = n_cols = K for a [N,K] weight)
        self.f.seek(self.data_start + off)
        # size: for Q4_K, n_elems = prod(dims); bytes = n_elems/256 * 144
        n_elems = 1
        for d in dims:
            n_elems *= d
        # (elems_per_block, bytes_per_block) per ggml type
        BLK = {8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
        if ttype in BLK:
            epb, bpb = BLK[ttype]; nbytes = n_elems // epb * bpb
        elif ttype == 1:   # F16
            nbytes = n_elems * 2
        elif ttype == 0:   # F32
            nbytes = n_elems * 4
        else:
            raise RuntimeError(f"unhandled ggml type {ttype} for {name}")
        return dims, ttype, self.f.read(nbytes)


def get_scale_min_k4(scbytes):
    """12 bytes -> (sc[8], mn[8]) six-bit, exactly marlin_gguf_ppu.cuh:gguf_q4k_scales."""
    q = np.frombuffer(scbytes, dtype=np.uint8)
    sc = np.zeros(8, np.int32); mn = np.zeros(8, np.int32)
    for g in range(4):
        sc[g] = q[g] & 63
        mn[g] = q[g + 4] & 63
    for g in range(4, 8):
        sc[g] = (q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4)
        mn[g] = (q[g + 4] >> 4)  | ((q[g - 0] >> 6) << 4)
    return sc, mn


def unpack_q4k_expert(raw_bytes, N, K):
    """Q4_K tensor bytes for one expert, logical [N][K] (K/256 blocks per row).
    Returns q_signed[K,N] int8, scale[K//32,N] f16, zero[K//32,N] f16 (native affine)."""
    nblk = K // 256
    blk = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N, nblk, 144)
    scale_k = K // 32
    q_signed = np.zeros((K, N), np.int8)
    scale = np.zeros((scale_k, N), np.float32)
    zero  = np.zeros((scale_k, N), np.float32)
    for n in range(N):
        for b in range(nblk):
            rec = blk[n, b]
            d = np.frombuffer(rec[0:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            dmin = np.frombuffer(rec[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
            sc, mn = get_scale_min_k4(rec[4:16].tobytes())
            qs = rec[16:144]  # 128 bytes = 256 nibbles
            for sub in range(8):
                sv = d * sc[sub]
                g = b * 8 + sub
                scale[g, n] = sv
                zero[g, n] = -dmin * mn[sub] + 8.0 * sv
                # nibbles for this sub-block: byte (sub/2)*32 + j, low if sub even else high
                base = (sub // 2) * 32
                for j in range(32):
                    byte = qs[base + j]
                    qv = (byte & 0xF) if (sub % 2 == 0) else (byte >> 4)
                    k = b * 256 + sub * 32 + j
                    q_signed[k, n] = int(qv) - 8
    return q_signed, scale.astype(np.float16), zero.astype(np.float16)


# =========================================================== GPTQ unpack (sym, desc_act=false)
def unpack_gptq_expert(qweight, qzeros, scales, gs, bits=4):
    """qweight I32[K/8,N], qzeros I32[K/gs, N/8], scales F16[K/gs, N].
    Returns q_signed[K,N] int8, scale[K/gs,N] f16 (scale-only for sym)."""
    assert bits == 4
    Kp8, N = qweight.shape
    K = Kp8 * 8
    scale_k = scales.shape[0]
    # unpack qweight: 8 nibbles per int32 along K, low-to-high
    qw = qweight.astype(np.uint32)
    q_u = np.zeros((K, N), np.int32)
    for s in range(8):
        q_u[s::8, :] = (qw >> (4 * s)) & 0xF
    # unpack qzeros (for verification): 8 per int32 along N. NOTE: this GPTQModel sym checkpoint stores the raw
    # zero-point directly (NO AutoGPTQ +1 offset) -- verified: raw unpacked zero == 8 AND dequant W mean ~= 0 with
    # q_signed = q_u - 8. (If it were the +1 convention the zero would read 7.) So symmetric zero = 8.
    zk, zn8 = qzeros.shape
    qz = qzeros.astype(np.uint32)
    zero_u = np.zeros((scale_k, N), np.int32)
    for s in range(8):
        zero_u[:, s::8] = (qz >> (4 * s)) & 0xF
    uniq = np.unique(zero_u)             # raw stored zero-point; sym -> all 8
    assert set(uniq.tolist()) <= {8}, f"scale-only path requires symmetric zero==8, but qzeros has {uniq}"
    q_signed = (q_u - 8).astype(np.int8)  # sym: subtract the (constant) zero-point 8 -> [-8,7]
    return q_signed, scales.astype(np.float16), uniq


# =========================================================== golden + dump
def golden_and_pack(q_signed, scale, zero, gs, M, seed=1234):
    """q_signed[K,N], scale[scale_k,N], zero or None. Returns A[M,K] f16, golden[M,N] f16."""
    K, N = q_signed.shape
    rng = np.random.default_rng(seed)
    # Larger A (integers +-7, std ~4) so D = A@W is O(1) for real (small-magnitude) weights -- keeps the match well
    # above the driver's 1e-2 abs floor, i.e. non-trivial (not a mostly-zero 0==0 pass).
    A = rng.integers(-7, 8, size=(M, K)).astype(np.float16)
    # dequant W[k,n] = scale[k//gs,n]*q_signed + (zero[k//gs,n] or 0)
    g = (np.arange(K) // gs)
    W = scale.astype(np.float32)[g, :] * q_signed.astype(np.float32)
    if zero is not None:
        W = W + zero.astype(np.float32)[g, :]
    golden = (A.astype(np.float32) @ W).astype(np.float16)   # fp32 accumulate
    return A, golden


def write_bin(path, L, M, N, K, gs, mode, A_list, qs_list, sc_list, zr_list, gold_list, bits=4):
    # q is computed/held as [K][N] (q[k][n]=weight(in k,out n)); the grouped kernel reads the raw B buffer as
    # [N][K] (q_buf[n*K+k] -> W(out n,in k)), so DUMP q TRANSPOSED to [N][K]. scale stays [scale_k][N]; golden
    # stays true physics. (Localized via the driver's 4-way host-golden probe: G3 q-transposed matched.)
    # bits: 4 -> q_signed in [-8,7] (W4A16, int4 slot); 8 -> [-128,127] (W8A16, int8 slot). q dumped int8 either way.
    with open(path, "wb") as f:
        f.write(b"RWMOE\0\0\0")
        f.write(struct.pack("<7i", L, M, N, K, gs, mode, bits))
        for A in A_list:   f.write(np.ascontiguousarray(A, dtype=np.float16).tobytes())
        for q in qs_list:  f.write(np.ascontiguousarray(q.T, dtype=np.int8).tobytes())   # [K][N] -> [N][K]
        for s in sc_list:  f.write(np.ascontiguousarray(s, dtype=np.float16).tobytes())
        if mode == 1:
            for z in zr_list: f.write(np.ascontiguousarray(z, dtype=np.float16).tobytes())
        for g in gold_list: f.write(np.ascontiguousarray(g, dtype=np.float16).tobytes())
    # golden magnitude stats -> guard against a trivial (mostly-zero) MATCH
    gall = np.concatenate([np.asarray(g, np.float32).ravel() for g in gold_list])
    a = np.abs(gall)
    print(f"[dump] wrote {path}: L={L} M={M} N={N} K={K} gs={gs} mode={mode}")
    print(f"[dump] golden stats: mean|.|={a.mean():.4g} max|.|={a.max():.4g} "
          f"nonzero(|.|>1e-2)={100.0*(a>1e-2).mean():.1f}%  (low nonzero%% => MATCH would be near-trivial)")


def parse_experts(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# =========================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gptq", "gguf", "synth"])
    ap.add_argument("path", nargs="?", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", default="0-7")
    ap.add_argument("--proj", default="gate", choices=["gate", "up", "down"])
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    # synth-only shape overrides
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--gs", type=int, default=128)
    ap.add_argument("--mode", type=int, default=0, choices=[0, 1])
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])  # synth: int4 (W4A16) or int8 (W8A16) slot
    args = ap.parse_args()
    experts = parse_experts(args.experts) if "-" in args.experts or "," in args.experts else list(range(int(args.experts)))

    if args.kind == "synth":
        # random q_signed[K][N] in [-8,7] + random scale[scale_k][N] (+zero if mode==1) + native golden.
        # Exercises the driver + golden orientation with NO real-weight extraction, to isolate harness vs kernel.
        N, K, gs, mode, bits = args.n, args.k, args.gs, args.mode, args.bits
        sk = (K + gs - 1) // gs
        qlo, qhi = (-8, 8) if bits == 4 else (-128, 128)
        rng = np.random.default_rng(2024)
        qs_list, sc_list, zr_list, A_list, gold_list = [], [], [], [], []
        L = len(experts)
        for e in range(L):
            q = rng.integers(qlo, qhi, size=(K, N)).astype(np.int8)
            sc = (0.02 + rng.integers(0, 8, size=(sk, N)) * 0.01).astype(np.float16)
            zr = ((rng.random((sk, N)).astype(np.float32) - 0.5) * 0.2).astype(np.float16) if mode == 1 else None
            A, gold = golden_and_pack(q, sc, zr, gs, args.m, seed=1000 + e)
            qs_list.append(q); sc_list.append(sc); A_list.append(A); gold_list.append(gold)
            if mode == 1: zr_list.append(zr)
        out = args.out or f"synth_gs{gs}_mode{mode}_int{bits}.bin"
        write_bin(out, L, args.m, N, K, gs, mode, A_list, qs_list, sc_list, zr_list, gold_list, bits=bits)
        return

    if args.kind == "gguf":
        import gguf_dequant as gq
        g = Gguf(args.path)
        cand = [f"blk.{args.layer}.ffn_{args.proj}_exps.weight",
                f"blk.{args.layer}.ffn_{args.proj}_exps.{args.proj}.weight"]
        tname = next((c for c in cand if c in g.tensors), None)
        if tname is None:
            print(f"[gguf] could not find expert tensor for proj={args.proj}; candidates {cand}")
            print("       available block-0 ffn tensors:")
            for k in sorted(g.tensors):
                if "blk.0." in k and "ffn" in k.lower():
                    print("        ", k, g.tensors[k][0], "type", g.tensors[k][1])
            sys.exit(1)
        dims, ttype, raw = g.raw(tname)
        fmt = gq.GGML_TYPE.get(ttype, f"type{ttype}")
        K, N, E = dims[0], dims[1], dims[2]                      # [K(in), N(out), n_expert] fastest-first
        epb = 32 if fmt == "q8_0" else 256
        per_expert_bytes = (K * N) // epb * gq.BLOCK_BYTES.get(fmt, 0)
        print(f"[gguf] {tname} dims={dims} ggml_type={ttype} ({fmt})  K(in)={K} N(out)={N} n_expert={E}")

        def unpack(rawe):   # dispatch: Q4_K -> this file's int4 unpacker; others -> gguf_dequant
            if ttype == GGML_Q4_K:
                q, sc, zr = unpack_q4k_expert(rawe, N, K); return q, sc, zr, 32, 4
            if fmt not in gq.UNPACK:
                print(f"[gguf] unsupported ggml_type {ttype} ({fmt})"); sys.exit(1)
            return gq.UNPACK[fmt](rawe, N, K)

        q0, sc0, zr0, gs, bits = unpack(raw[:per_expert_bytes])
        mode = 0 if zr0 is None else 1
        print(f"[gguf] fmt={fmt} gs={gs} slot=int{bits} mode={mode} per_expert_bytes={per_expert_bytes}")
        if args.selfcheck:
            Wf = sc0.astype(np.float32)[(np.arange(K) // gs), :] * q0.astype(np.float32)
            if zr0 is not None: Wf += zr0.astype(np.float32)[(np.arange(K) // gs), :]
            print(f"[selfcheck] expert0 {args.proj} {fmt}: q_signed range [{q0.min()},{q0.max()}] "
                  f"scale[min,max]=[{sc0.astype(np.float32).min():.4g},{sc0.astype(np.float32).max():.4g}] "
                  f"W[min,max,mean,std]=[{Wf.min():.4g},{Wf.max():.4g},{Wf.mean():.4g},{Wf.std():.4g}]")
            return
        qs_list, sc_list, zr_list, A_list, gold_list = [], [], [], [], []
        for e in experts:
            q, sc, zr, _, _ = unpack(raw[e * per_expert_bytes:(e + 1) * per_expert_bytes])
            A, gold = golden_and_pack(q, sc, zr, gs, args.m, seed=1000 + e)
            qs_list.append(q); sc_list.append(sc); A_list.append(A); gold_list.append(gold)
            if mode == 1: zr_list.append(zr)
        out = args.out or f"real_{fmt}_{args.proj}_L{args.layer}.bin"
        write_bin(out, len(experts), args.m, N, K, gs, mode, A_list, qs_list, sc_list, zr_list, gold_list, bits=bits)

    else:  # gptq
        # config.json / index.json may live in --path OR a parent dir (safetensors can be in a subdir like ow7_224_ca).
        def find_up(start, name, levels=4):
            d = os.path.abspath(start)
            for _ in range(levels):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return p
                nd = os.path.dirname(d)
                if nd == d:
                    break
                d = nd
            return None
        cfg_path = find_up(args.path, "config.json")
        idx_path = find_up(args.path, "model.safetensors.index.json")
        assert cfg_path, f"config.json not found at/above {args.path}"
        assert idx_path, f"model.safetensors.index.json not found at/above {args.path}"
        shard_dir = os.path.dirname(idx_path)   # shards live next to the index
        print(f"[gptq] config={cfg_path}\n[gptq] index={idx_path}\n[gptq] shard_dir={shard_dir}")
        cfg = json.load(open(cfg_path))
        qc = cfg.get("quantization_config", {})
        bits = qc.get("bits", 4); gs = qc.get("group_size", 128); sym = qc.get("sym", True)
        desc_act = qc.get("desc_act", False)
        print(f"[gptq] bits={bits} gs={gs} sym={sym} desc_act={desc_act}")
        assert bits == 4 and sym and not desc_act, "this path assumes bits=4, sym, desc_act=false"
        idx = json.load(open(idx_path))["weight_map"]
        # resolve prefix (model.language_model.layers.{L}.mlp.experts.{E}.{proj}_proj)
        def key(e, sfx):
            return f"model.language_model.layers.{args.layer}.mlp.experts.{e}.{args.proj}_proj.{sfx}"
        # open the needed shards lazily
        open_shards = {}
        def load(e, sfx):
            k = key(e, sfx)
            shard = idx[k]
            if shard not in open_shards:
                open_shards[shard] = st_open(os.path.join(shard_dir, shard))
            f, hdr, ds = open_shards[shard]
            return st_get(f, hdr, ds, k)
        qs_list, sc_list, A_list, gold_list = [], [], [], []
        K = N = None
        for e in experts:
            qw = load(e, "qweight"); qz = load(e, "qzeros"); scl = load(e, "scales")
            q, sc, zp_uniq = unpack_gptq_expert(qw, qz, scl, gs, bits)
            if e == experts[0]:
                K, N = q.shape
                print(f"[gptq] K(in)={K} N(out)={N} scale_k={sc.shape[0]}  zero_point unique(+1)={zp_uniq}")
            A, gold = golden_and_pack(q, sc, None, gs, args.m, seed=1000 + e)
            qs_list.append(q); sc_list.append(sc); A_list.append(A); gold_list.append(gold)
        out = args.out or f"real_gptq_{args.proj}_L{args.layer}.bin"
        write_bin(out, len(experts), args.m, N, K, gs, 0, A_list, qs_list, sc_list, None, gold_list)


if __name__ == "__main__":
    main()
