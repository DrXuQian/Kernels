# Marlin FP16xINT4 GEMM - PPU experiments

Standalone Marlin W4A16 GEMM experiments for the T-Head PPU (ppu001 / ACOMPUTE 10000). The current path keeps the classic Marlin structure and patches only the PPU-specific parts.

## Current status

- `marlin_classic_ppu.cuh`: the main classic Marlin PPU port.
  - Fuses two NVIDIA `m16n8k16` MMAs into one PPU `m16n16k16` MMA.
  - Uses the verified PPU C fragment layout for result stores and global reduce.
  - `write_result` stages the output tile through shared memory in row-major order and streams it out as `int4`
    (8 halves per store). Worth ~1% end-to-end, not more: the epilogue runs once per CTA while the mainloop runs
    `k_tiles` iterations, so the 8x drop in store *instructions* only ever bought the ~2.8% of runtime that the
    scattered writes actually cost in DRAM traffic. Kept for the cleaner epilogue, not for speed. The PPU C layout (`col = lane%4 + 4*(l%4)`) makes a lane's columns stride-4, so the direct
    store issued one 2-byte global store per accumulator float. `-DMARLIN_WRITE_DIRECT` restores that old path for
    A/B bisection. The row stride carries a `+1 int4` pad, which drops the staging bank conflicts from 16-way to 2-way.
  - Keeps the classic cp.async pipeline, dispatcher, dequant path, and K-warp reduction structure.
- `marlin_ppu.cuh`: a smaller plain-CUDA PPU kernel used as a self-consistent correctness prototype.
- Reference code for other backends has been removed from this directory; the remaining targets are PPU-only.

## Files

- `marlin_classic_ppu.cuh`: classic Marlin PPU port.
- `marlin_ppu.cuh`: simple standalone PPU prototype.
- `dequant_smoke.cu`: INT4 `lop3` dequant smoke test vs CPU.
- `dump_ldmatrix.cu`: ground truth for `ppu.ldmatrix.x4` register distribution.
- `test_marlin_ppu.cu`: self-consistent plain PPU kernel correctness test.
- `test_marlin_classic_ppu.cu`: classic port compile/run gate.
- `test_marlin_classic_num.cu`: classic port numerical test for the non-split path.
- `test_marlin_classic_splitk.cu`: classic split-K / global_reduce numerical test.
- `bench_marlin.cu`: performance bench for the classic PPU port.

## Build

Build on the PPU box only. macOS does not have the PPU nvcc/toolchain.

```sh
make NVCC=<ppu-nvcc> dequant_smoke
make NVCC=<ppu-nvcc> marlin_classic_num
make NVCC=<ppu-nvcc> marlin_classic_splitk
make NVCC=<ppu-nvcc> bench_marlin
```

PPU target is ppu001: do not pass `-arch`. A forced `sm_XX` target can route to ppu0015 and reject ppu001-only asm.

## Useful runs

```sh
./marlin_classic_num
./marlin_classic_splitk
./bench_marlin
./bench_marlin 2048 4096 14336
```

For occupancy/register sweeps, keep `MARLIN_MIN_BLOCKS` as a build-time override:

```sh
F="-O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda"
for mb in 2 3 4; do
  make clean >/dev/null
  make NVCCFLAGS="$F -DMARLIN_MIN_BLOCKS=$mb -Xptxas -v" bench_marlin
  ./bench_marlin 2048 4096 14336
done
```

Current observation: `MARLIN_MIN_BLOCKS=2` was best in the `2048 x 4096 x 14336` sweep because avoiding register spills beat the extra theoretical occupancy.

`MARLIN_MAX_MB=2` is also the measured optimum. Raising it to 3 amortizes dequant over 3 MMAs instead of 2, but the
`(3,16,4)` kernel spills 23-32 VRegs for real, and the spills cost more than the saved `lop3`s:

| shape | `MAX_MB=2` | `MAX_MB=3` |
|---|---|---|
| 2048 x 4096 x 14336 | 292.8 TFLOP/s | 231.5 |
| 4096 x 4096 x 4096 | 262.9 TFLOP/s | 231.4 |

So dequant amortization is not worth register pressure here; do not spend effort splitting `frag_c` to reach `MB=4`.

## Known caveats

- `bench_marlin` uses random data and is for performance only; correctness is validated separately by `test_marlin_classic_num` and `test_marlin_classic_splitk`.
- `write_result` and `global_reduce` use the PPU C fragment layout directly.
