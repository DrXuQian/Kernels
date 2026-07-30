// GENERATED-INCLUDED per-unit body for the GEMV perf sweep. One unit = one (format, layout, StepK, Threads,
// CtaN, Chunk). The generator defines UNIT_* and includes this.
//
// The body must be a TEMPLATE. `if constexpr (false)` around a call does NOT suppress instantiation in a plain
// function -- only inside a templated entity -- and an illegal (StepK, Threads) pair would then trip a
// static_assert in a unit that is never meant to run. Same trap the MoE sweep's unit body documents.
#include "gemv_perf_common.hpp"

template <int Dummy>
static void unit_body(Shape const& sh, Bufs const& b, Best& best) {
  using D = KernelDetails<FP16DetailsA, UNIT_FMT, UNIT_LAY, UNIT_SK, UNIT_TH>;
  run_row<D, UNIT_CN, UNIT_CH>(sh, b, best);
}

void UNIT_FN(Shape const& sh, Bufs const& b, Best& best) { unit_body<0>(sh, b, best); }
