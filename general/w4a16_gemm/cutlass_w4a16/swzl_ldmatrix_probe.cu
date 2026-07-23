// Standalone ppu ldmatrix.swzl ground-truth probe [box-only; needs hgcc/__HGGC_ARCH__==100]. No cutlass.
// Fills smem with tagged b16 words (word[i]=i), runs the REAL swzl ldmatrix (same asm as
// PPU0010_TSM_LD_SWZL_IMPL<int8_t,false>), and dumps each lane's 4 regs (= 8 b16 word-indices) -> the exact
// smem-word -> (lane, reg) map. Runs several coord_h to see the N (row) addressing: if coord_h=0 and coord_h=8
// read OVERLAPPING words, the swzl aliases the 8-N-halves (root of sigma_n=n&~8). This is the ground truth to
// derive the int2 interleave from.
//
// Build (box): TARGET=swzl_ldmatrix_probe ./build.sh   (or hgcc directly; needs PPU SDK). Run: ./<bin>
#include <cstdio>
#include <cstdint>

// CUBE_H = Block_MN (N-tile) = 64 in the real W2A16 config. channel_bytes_offset = coord_w (K byte offset) = 0.
template <int CUBE_H>
__global__ void probe(int coord_h) {
  __shared__ uint16_t smem[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem[i] = (uint16_t)i;   // tag: b16 word i holds value i
  __syncthreads();

  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  void* base = (void*)smem;
  asm volatile(
    "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "l"(base), "r"(0), "r"(coord_h), "r"(1), "r"(CUBE_H), "r"(1), "r"(0));
#endif
  int lane = threadIdx.x % 32;
  // each reg = b32 = 2 b16 words; print the (lo,hi) word-indices that landed in each of the 4 regs
  printf("ch=%d lane %2d: r0(%u,%u) r1(%u,%u) r2(%u,%u) r3(%u,%u)\n",
         coord_h, lane, r0 & 0xffff, r0 >> 16, r1 & 0xffff, r1 >> 16,
         r2 & 0xffff, r2 >> 16, r3 & 0xffff, r3 >> 16);
}

int main() {
  printf("[swzl-probe] CUBE_H=64; word[i]=i; dump (lane,reg)->smem b16-word index. Compare coord_h=0 vs 8.\n");
  for (int ch : {0, 8}) { probe<64><<<1, 32>>>(ch); hggcDeviceSynchronize(); printf("----\n"); }
  return 0;
}
