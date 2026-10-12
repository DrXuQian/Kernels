# PPU 1.0 AIU read benchmark

Base: Kernels `0836d02c40b6d41708f001a670b97eef3ee879fc`.
Scope: a standalone PPU 1.0 (ZW810) HBM-to-shared read ceiling experiment.
No model, GEMM, dequantization, global output copy, or production routing edits.

- Move raw bytes with the native AIU linear 2D bulk-copy instruction.
- Cover the input exactly once per launch using a CTA grid-stride traversal.
- Vary tile size (4/8/16 KiB), outstanding groups (1/2/4/8), and grid size
  (1/2/4/8 CTAs per reported compute unit); retain every tested result.
- Validate every copied word outside timing against address-dependent integer
  tags, including pipeline wrap and drain; verify the timed kernel's final
  per-CTA sentinels too. Do not time a failing configuration.
- Default: 2 GiB source, at least 16 times the verified L2/LLC capacity.
  Read a separate 4-times-cache eviction allocation before every sample,
  outside the measured interval. Reject an unknown cache size without an
  explicit operator-supplied capacity.
- Exclude initialization, validation, eviction and two warmup launches.
  Report individual event samples, median/min/max, requested bytes / time,
  and optional percent of an explicitly supplied bandwidth peak.
- Inspect the compiled PPU 1.0 ELF for native AIU loads and completion waits.
  Compilation is not device validation; effective requested-byte bandwidth
  is not an ACU measurement of actual DRAM transactions.

Local delivery gate: compile, native ISA review, host schedule self-test,
argument checks, clean source-only commit. PPU word checking, resource limits,
timing and ACU DRAM counters are the box gate; no HBM result is claimed locally.
Bounded first run: 48 configurations, no adaptive or unbounded tuning loop.
