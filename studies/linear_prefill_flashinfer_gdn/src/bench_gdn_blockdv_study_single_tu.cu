// Single translation unit build for the study-only block-DV GDN prefill path.
//
// This mirrors bench_gdn_tile_study_single_tu.cu so SM90 warp-specialized
// kernels can keep setmaxnreg instead of falling back to serialized WGMMA.

#include "gdn_blockdv_study_dispatch.cu"

#ifdef CHECK
#undef CHECK
#endif

#include "bench_gdn_blockdv_study.cu"
