// Single translation unit build for FlashInfer GDN prefill study.
//
// The normal study target mirrors the production-style separable compilation
// (`-dc`) flow. This target includes dispatch and main in one TU so ptxas can
// honor warpgroup register reallocation for the SM90 warp-specialized kernel.

#include "gdn_tile_study_dispatch.cu"

#ifdef CHECK
#undef CHECK
#endif

#include "bench_gdn_tile_study.cu"
