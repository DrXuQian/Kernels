# Resume

Local SDK 2.1.1 compilation passed. All 12 timed template bodies contain one
native `vmem.aiu.ld.tsm...2d.b8` site, one commit site, the selected outstanding
group wait and final drain. They use 32 vector registers, zero stack, no CTA
barriers and no MMA. The host tile/slot model passed 288 cases. These are
compile/host checks, not hardware admission. The local host lacks the PPU
runtime library and cannot execute the GPU gates.

Native device correctness and HBM bandwidth have not been measured. Run only on an otherwise
idle PPU; do not interpret another model's concurrent traffic as this kernel's
limit. See README.md for build/run/profile commands and docs/plan.md for the
measurement contract. Preserve stdout, binary hash, SDK identity and ACU report.
