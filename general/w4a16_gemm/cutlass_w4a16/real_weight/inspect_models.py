#!/usr/bin/env python3
"""Inspect the real Qwen3.5 GPTQ-Int4 (safetensors) and Q4_K_M (GGUF) checkpoints so we can write the
real-weight extractor against the ACTUAL tensor names / shapes / quant config (rather than guessing).

Run on the PPU box (needs: numpy, safetensors). It does NOT need torch or the model loaded on GPU.

    python3 inspect_models.py gptq /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-122B-A10B-GPTQ-Int4/ow7_224_ca
    python3 inspect_models.py gguf /sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf

Paste the output back and I'll finalize dump_real_weights.py + test_moe_grouped_real.cu.
"""
import sys, os, json, struct, glob


# ----------------------------------------------------------------------------- GPTQ (safetensors)
def inspect_gptq(path):
    print(f"=== GPTQ inspect: {path} ===")
    # config + quant config
    for cfg_name in ("config.json", "quantize_config.json", "quant_config.json"):
        p = os.path.join(path, cfg_name)
        if os.path.exists(p):
            with open(p) as f:
                cfg = json.load(f)
            print(f"\n--- {cfg_name} ---")
            # print the keys most relevant to quant + MoE shape
            interesting = ("bits", "group_size", "sym", "desc_act", "quant_method", "checkpoint_format",
                           "hidden_size", "intermediate_size", "moe_intermediate_size", "num_experts",
                           "num_experts_per_tok", "n_routed_experts", "num_hidden_layers", "model_type",
                           "architectures", "shared_expert_intermediate_size")
            for k in interesting:
                if k in cfg:
                    print(f"  {k}: {cfg[k]}")
            if "quantization_config" in cfg:
                print(f"  quantization_config: {json.dumps(cfg['quantization_config'])[:500]}")
        else:
            continue

    # safetensors: read each file's JSON header (8-byte LE length + JSON) without loading tensors.
    sts = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    print(f"\n--- {len(sts)} safetensors file(s) ---")
    if not sts:
        print("  (none found)")
        return
    # aggregate keys across shards; show a representative expert-0 layer-0 set
    all_keys = {}
    for st in sts:
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode("utf-8"))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            all_keys[k] = (v.get("dtype"), v.get("shape"))
    print(f"  total tensors: {len(all_keys)}")

    # show all distinct key TEMPLATES (strip layer/expert indices) with one example + dtype/shape
    import re
    templ = {}
    for k, (dt, shp) in all_keys.items():
        t = re.sub(r"\.\d+\.", ".{}.", k)
        if t not in templ:
            templ[t] = (k, dt, shp)
    print("\n  distinct tensor-name templates (example | dtype | shape):")
    for t in sorted(templ):
        ex, dt, shp = templ[t]
        print(f"    {t}")
        print(f"       e.g. {ex}  {dt}  {shp}")

    # zoom into MoE expert quant tensors for layer 0, expert 0 (best-effort name search)
    print("\n  MoE-expert-like quant tensors (grep 'expert' + qweight/qzeros/scales/g_idx):")
    for k in sorted(all_keys):
        kl = k.lower()
        if ("expert" in kl) and any(s in kl for s in ("qweight", "qzeros", "scales", "g_idx", "scale")):
            if ".0." in k or "layers.0" in k or "experts.0" in k:  # keep it short: layer/expert 0-ish
                dt, shp = all_keys[k]
                print(f"    {k}  {dt}  {shp}")


# ----------------------------------------------------------------------------- GGUF
GGUF_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
             9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
             15: "Q8_K", 16: "IQ2_XXS", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16"}
# metadata value types
_GT = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
       8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
        10: "<Q", 11: "<q", 12: "<d"}
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _rd_val(f, vt):
    if vt == 8:
        return _rd_str(f)
    if vt == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            return [_rd_str(f) for _ in range(n)]
        vals = [struct.unpack(_FMT[et], f.read(_SZ[et]))[0] for _ in range(min(n, 8))]
        # skip the rest
        if n > 8:
            f.seek((n - 8) * _SZ[et], 1)
        return vals + (["...(%d total)" % n] if n > 8 else [])
    return struct.unpack(_FMT[vt], f.read(_SZ[vt]))[0]


def inspect_gguf(path):
    print(f"=== GGUF inspect: {path} ===")
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"bad magic {magic!r}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"  version={version} n_tensors={n_tensors} n_kv={n_kv}")

        print("\n--- metadata (arch / moe / quant relevant) ---")
        kv = {}
        for _ in range(n_kv):
            key = _rd_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            val = _rd_val(f, vt)
            kv[key] = val
        for k in sorted(kv):
            kl = k.lower()
            if any(s in kl for s in ("architecture", "expert", "block_count", "embedding_length",
                                     "feed_forward", "context_length", "quant", "alignment",
                                     "attention.head", "vocab")):
                print(f"    {k} = {kv[k]}")

        # tensor infos
        infos = []
        for _ in range(n_tensors):
            name = _rd_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            infos.append((name, dims, ttype, offset))

        print(f"\n--- {len(infos)} tensors: distinct name templates (example | ggml_type | dims) ---")
        import re
        templ = {}
        for name, dims, ttype, off in infos:
            t = re.sub(r"\.\d+\.", ".{}.", name)
            t = re.sub(r"blk\.\d+", "blk.{}", t)
            if t not in templ:
                templ[t] = (name, ttype, dims)
        for t in sorted(templ):
            ex, ttype, dims = templ[t]
            print(f"    {t}   [{GGUF_TYPE.get(ttype, '?%d' % ttype)}]  dims={dims}   e.g. {ex}")

        # zoom into block 0 ffn/expert tensors
        print("\n--- block 0 ffn/expert tensors (name | ggml_type | dims) ---")
        for name, dims, ttype, off in infos:
            nl = name.lower()
            if ("blk.0." in name or ".0." in name) and ("ffn" in nl or "exp" in nl):
                print(f"    {name}  [{GGUF_TYPE.get(ttype, '?%d' % ttype)}]  dims={dims}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    kind, path = sys.argv[1], sys.argv[2]
    if kind == "gptq":
        inspect_gptq(path)
    elif kind == "gguf":
        inspect_gguf(path)
    else:
        print("first arg must be 'gptq' or 'gguf'")
        sys.exit(1)
