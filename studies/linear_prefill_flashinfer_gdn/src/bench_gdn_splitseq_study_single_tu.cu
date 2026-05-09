// Single-translation-unit build for the split-seq GDN study.
// This avoids separable device linking losing SM90 warp-specialized attributes
// on some CUDA/PPU toolchains.

#include "gdn_splitseq_study_dispatch.cu"
#include "gdn_state_only_study_dispatch.cu"

#define main gdn_splitseq_study_main_impl
#include "bench_gdn_splitseq_study.cu"
#undef main

int main(int argc, char** argv) {
  return gdn_splitseq_study_main_impl(argc, argv);
}
