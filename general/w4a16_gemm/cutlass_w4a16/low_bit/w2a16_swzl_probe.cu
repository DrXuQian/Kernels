// W2A16 swzl map probe — empirically dumps the byte/word-level ldmatrix.swzl layout by running actlize's OWN
// software model (ppu_tsm_ld_swzl_sim from copy_ppu0010_aiu.hpp), copied here self-contained so it builds on any
// CUDA GPU (the sim is plain arithmetic + pointer reads, NO ppu asm). Purpose: confirm the b16-level reshuffle
// baked into MixGemmNumericArrayConverter<half_t,uint2b_t,64> (VREG order + b16 swaps). The swzl is IDENTICAL for
// int4/int2 (both drive PPU0010_TSM_LD_SWZL<int8_t>; the map depends on CUBE_H/lane/coord, not sub-byte width),
// so this word-level map governs both. The WITHIN-b16 (8 uint2 per b16 -> mma fp16) order is NOT covered here
// (that's the fp16 mma fragment, needs the box); this probe only nails the coarse word/vreg reshuffle.
//
//   nvcc -std=c++17 -O3 w2a16_swzl_probe.cu -o w2a16_swzl_probe && ./w2a16_swzl_probe
//
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

// --- verbatim logic from ppu_tsm_ld_swzl_sim<Element,CUBE_H,CUBE_W,SWAP=false>, SZOF_ELEM = sizeof(Element) ---
// Element = int8_t (what the uint2/int4 operand passes to the swzl) => SZOF_ELEM = 1.
template<int SZOF_ELEM, int CUBE_H, int CUBE_W>
__device__ void swzl_sim(uint32_t* frag, const uint32_t* stage_base_u32, int coord_w, int coord_h, int stage) {
  // original: Element* stage_base = smem; stage_base += CUBE_H*CUBE_W*stage; slice_base=(u32*)stage_base;
  const uint32_t* slice_base = stage_base_u32 + (CUBE_H * CUBE_W * stage) / (4 / SZOF_ELEM);
  int slice_idx = coord_w / (32 / SZOF_ELEM);
  slice_base += CUBE_H * 8 * slice_idx;                         // uint32 units
  int slice_start_vec = (((slice_idx & 1) << 1) + ((slice_idx & 2) >> 1)) * 2;
  int lane_idx = threadIdx.x % 32;
  int lane_row_idx = lane_idx / 4 + coord_h;                    // warp arranged 8x4
  int lane_col_idx = lane_idx % 4;
  for (int v = 0; v < 4; v++) {
    int vreg_row_idx  = (v % 2) * 8 + lane_row_idx;             // SWAP=false: vreg 1/3 on lower 8 rows
    int vreg_line_idx = vreg_row_idx / 4;
    int vreg_vec_idx  = (vreg_row_idx % 4) * 2 + (v / 2);       // SWAP=false: vreg 2/3 on right
    int vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
    int vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
    int lane_32b_off = vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
    frag[v] = slice_base[lane_32b_off];                         // tagged: word i holds value i -> frag=word idx
  }
}

template<int CUBE_H, int CUBE_W>
__global__ void probe(const uint32_t* buf, uint32_t* out, int coord_w, int coord_h, int stage) {
  uint32_t frag[4];
  swzl_sim<1, CUBE_H, CUBE_W>(frag, buf, coord_w, coord_h, stage);
  int lane = threadIdx.x % 32;
  for (int v = 0; v < 4; ++v) out[lane * 4 + v] = frag[v];
}

int main() {
  const int CUBE_H = 16, CUBE_W = 64;     // representative (Block_MN=16, AiuContElemSize=64)
  const int NW = 4096;                    // tagged word buffer (word i = i)
  uint32_t *dbuf, *dout; std::uint32_t hbuf[NW]; for (int i = 0; i < NW; ++i) hbuf[i] = i;
  CK(cudaMalloc(&dbuf, NW * 4)); CK(cudaMemcpy(dbuf, hbuf, NW * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dout, 32 * 4 * 4));

  auto dump = [&](int coord_w, int coord_h, int stage) {
    probe<CUBE_H, CUBE_W><<<1, 32>>>(dbuf, dout, coord_w, coord_h, stage);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint32_t h[128]; CK(cudaMemcpy(h, dout, 32 * 4 * 4, cudaMemcpyDeviceToHost));
    printf("== swzl word-map  CUBE_H=%d CUBE_W=%d  coord_w=%d coord_h=%d stage=%d ==\n", CUBE_H, CUBE_W, coord_w, coord_h, stage);
    printf("lane : vreg0 vreg1 vreg2 vreg3   (each = smem uint32 word index landed in that vreg)\n");
    for (int l = 0; l < 32; ++l)
      printf("%3d  : %5u %5u %5u %5u\n", l, h[l*4+0], h[l*4+1], h[l*4+2], h[l*4+3]);
    return 0;
  };
  dump(0, 0, 0);
  return 0;
}
