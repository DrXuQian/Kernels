# Model config shadows

These files are small `config.json` snapshots used to keep benchmark wrapper
shape choices auditable without downloading model weights.

| File | Source | Notes |
|---|---|---|
| `Qwen3.5-27B.config.json` | `Qwen/Qwen3.5-27B` | Dense fp16 wrapper source. |
| `Qwen3.5-122B-A10B-GPTQ.config.json` | `Qwen/Qwen3.5-122B-A10B` fallback | The GPTQ raw config endpoint returned 401 in this environment; the non-GPTQ config has the same text model shape used by the GPTQ TP wrapper. |
| `MiniMax-M2.7.config.json` | `MiniMaxAI/MiniMax-M2.7` | FP8 E4M3 block-wise weight-quantized model; `gate` and `lm_head` are excluded from quantization. |
