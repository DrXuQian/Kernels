// Ground truth for `ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16`: which (row,col) of a 16x16 half tile lands in
// which of the 4 output registers, per lane. Fills shared with val = row*16+col (0..255, exact in fp16), has each
// lane supply Marlin's a_sh_rd address, then decodes what came back.
//
// Reference points printed alongside:
//   * the PPU mma A operand wants reg l = (row frag_i(lane,l), k 2*frag_j(lane,l) .. +1)
//   * NVIDIA ldmatrix.x4 returns the 4 matrices in lane-address order: v_m = the tile addressed by lanes 8m..8m+7
//
// build (box):  make dump_ldmatrix   (nvcc -O3, no -arch).  Run: ./dump_ldmatrix

#include <cuda_fp16.h>
#include <cstdio>

#define K 16                      // halves per row of the tile
#define ROWS 16

__device__ __forceinline__ int frag_i(int lane, int l) { return (lane / 4) + (l / 2) * 8; }
__device__ __forceinline__ int frag_j(int lane, int l) { return (lane % 4) + (l % 2) * 4; }

__global__ void dump() {
    __shared__ __half sm[ROWS * K];
    if (threadIdx.x == 0)
        for (int r = 0; r < ROWS; ++r)
            for (int c = 0; c < K; ++c) sm[r * K + c] = __float2half((float) (r * K + c));
    __syncthreads();

    const int lane = threadIdx.x;
    const __half * p = &sm[(lane % 16) * K + (lane / 16) * 8];   // Marlin's a_sh_rd addressing
    uint32_t v[4];
    asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(p));

    for (int w = 0; w < 32; ++w) {                       // serialize so the print is ordered
        if (lane == w) {
            printf("lane %2d | ", lane);
            for (int i = 0; i < 4; ++i) {
                const __half2 h = *reinterpret_cast<const __half2*>(&v[i]);
                int lo = (int) __half2float(__low2half(h)), hi = (int) __half2float(__high2half(h));
                printf("v%d=(%d,%2d)(%d,%2d) ", i, lo / K, lo % K, hi / K, hi % K);
            }
            printf("|| mma wants ");
            for (int l = 0; l < 4; ++l) printf("a%d=(%d,%2d)(%d,%2d) ", l, frag_i(lane, l), 2 * frag_j(lane, l),
                                               frag_i(lane, l), 2 * frag_j(lane, l) + 1);
            printf("\n");
        }
        __syncthreads();
    }
}

int main() {
    printf("ppu.ldmatrix.x4 dump: shared[r][c] = r*16+c; lane addr = &sm[(lane%%16)*16 + (lane/16)*8]\n");
    printf("each entry is (row,col) of the two halves packed in that register\n\n");
    dump<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("err=%s\n", cudaGetErrorString(e)); return 2; }
    printf("\nRead it as a permutation: if v_p == a_l, then frag_a[l] must be assigned v[p].\n");
    return 0;
}
