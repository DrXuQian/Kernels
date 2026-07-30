// L76 -- does PPU_A_CUBE_H actually give partition_S an M mode? Reported by the compiler, front end only.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/config/gemm_operands.hpp"
using namespace cute;
template <int TM, int TK, int CH> struct P {
  using Op = cutlass::gemm::config::DefaultGemm_AIU_Operand<
      cutlass::arch::PPU0010, cutlass::half_t, false, cute::Int<TM>, cute::Int<TK>, false, 0, true, CH>;
  static_assert(Op::CUBE_H == -1 && Op::CUBE_W == -1, "REPORT");
};
template struct P<16,256,0>;   // default: CUBE_H should be 16
template struct P<16,256,1>;   // override: CUBE_H should be 1
