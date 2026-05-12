// Single translation unit build for the production FlashInfer GDN prefill bench.
//
// The separable device compilation flow can make ptxas ignore SM90 setmaxnreg
// metadata for this warp-specialized kernel. Including dispatch and main in one
// TU keeps warpgroup register reallocation visible to ptxas.

#include "dispatch.cu"

#ifdef CHECK
#undef CHECK
#endif

#include "bench_gdn_prefill.cu"
