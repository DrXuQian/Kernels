// Can Marlin-format INT4 weights go through the AIU bulk load + swzl ldmatrix, the way the fp16 MoE GEMM
// moves bf16? This is the gate that decides whether the Q4_K MoE kernel is written on the AIU path or
// stays a derivation of the dense Marlin.
//
// THE QUESTION, precisely. Both the AIU bulk load and ldmatrix.swzl work in b16 (2-byte) units. Marlin's B
// is a stream of int32 words, and its dequant expects a lane to hold FOUR CONSECUTIVE int32 (= 8 b16). But
// ldmatrix .x4 hands a lane one 32-bit register from each of FOUR DIFFERENT 8x8 matrices -- not 8 adjacent
// elements. So the two layouts do not coincide by default, and the whole idea rests on whether we can
// PRE-PERMUTE B (a load-time repack we already do anyway) so that what swzl delivers to lane L is exactly
// the int4 Marlin wants. There is precedent -- the FA port made "swzl frag == ggml tile" -- but precedent
// is not proof for a different element width, so measure it.
//
// STAGE 1 dumps the map. Every b16 in the cube gets a unique id, so reading a lane's registers back says
// exactly which (h, w) each one came from. A bijection over the cube is what makes stage 2 possible at all;
// if the map skips or duplicates positions, no host permutation can fix it and the AIU path is closed for
// INT4.
// STAGE 2 uses the measured map to build a permuted B and checks a lane gets Marlin's four int32 in order.
// That is the real claim: not "the data arrives" but "it arrives in the arrangement dequant() needs".
//
// Build (box, ppu001, NO -arch): make aiu_int4_smoke && ./aiu_int4_smoke
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static __device__ __forceinline__ unsigned cvta(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }

// AIU asm, verbatim from the fp16 MoE GEMM / fa_ppu.cu. .b16 is dtype-agnostic: it moves 2-byte units.
static __device__ __forceinline__ void aiu_load(unsigned smem, const void* gmem,
        int dim_h, int dim_w, int cube_h, int cube_w, int coord_h, int coord_w) {
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem), "l"(gmem), "r"(1), "r"(dim_h), "r"(dim_w),
           "r"(0), "r"(coord_h), "r"(coord_w), "r"(1), "r"(cube_h), "r"(cube_w), "r"(1));
}
static __device__ __forceinline__ void ld_swzl(int* v, const void* cube_base, int coord_h, int coord_w, int cube_h) {
    const int cbo = coord_w * (int) sizeof(unsigned short);
    asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8, %9, %10};\n"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
        : "l"(cube_base), "r"(0), "r"(coord_h), "r"(1), "r"(cube_h), "r"(1), "r"(cbo));
}

#define CUBE_H 64
#define CUBE_W 64

// Each lane reports the 8 b16 ids it received, for one (coord_h, coord_w) of the cube.
__global__ void probe(const unsigned short* __restrict__ G, unsigned short* __restrict__ out,
                      int dim_h, int dim_w, int coord_h, int coord_w) {
    extern __shared__ unsigned short sh[];
    if (threadIdx.x == 0) aiu_load(cvta(sh), G, dim_h, dim_w, CUBE_H, CUBE_W, 0, 0);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (threadIdx.x >= 32) return;
    int v[4];
    ld_swzl(v, sh, coord_h, coord_w, CUBE_H);
    const unsigned short* p = reinterpret_cast<const unsigned short*>(v);
    for (int i = 0; i < 8; i++) out[threadIdx.x * 8 + i] = p[i];
}

int main() {
    printf("AIU + swzl ldmatrix on INT4-width data: does Marlin's per-lane int4 survive the path?\n");
    const int dim_h = CUBE_H, dim_w = CUBE_W;
    std::vector<unsigned short> hG((size_t) dim_h * dim_w);
    for (int h = 0; h < dim_h; h++)
        for (int w = 0; w < dim_w; w++) hG[(size_t) h * dim_w + w] = (unsigned short) (h * dim_w + w);

    unsigned short *dG, *dOut;
    CK(cudaMalloc(&dG, hG.size() * 2)); CK(cudaMalloc(&dOut, 32 * 8 * 2));
    CK(cudaMemcpy(dG, hG.data(), hG.size() * 2, cudaMemcpyHostToDevice));
    const size_t shmem = (size_t) CUBE_H * CUBE_W * 2;

    // ---- STAGE 1: the map, at a few (coord_h, coord_w) so a coordinate-dependent shift cannot hide ----
    bool bijective_all = true;
    for (int ch : {0, 16}) for (int cw : {0, 16}) {
        probe<<<1, 32, shmem>>>(dG, dOut, dim_h, dim_w, ch, cw);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned short> got(32 * 8);
        CK(cudaMemcpy(got.data(), dOut, got.size() * 2, cudaMemcpyDeviceToHost));

        std::vector<int> seen(dim_h * dim_w, 0);
        bool bij = true;
        for (int i = 0; i < 32 * 8; i++) { if (++seen[got[i]] > 1) bij = false; }
        bijective_all &= bij;
        printf("\n  coord=(%2d,%2d): %s over the 256 elements it touched\n", ch, cw,
               bij ? "BIJECTIVE" : "NOT bijective -- a host permutation CANNOT fix this");
        printf("    lane 0 regs: ");
        for (int i = 0; i < 8; i++) { const int id = got[i]; printf("(%d,%d)%s", id / dim_w, id % dim_w, i == 7 ? "\n" : " "); }
        printf("    lane 1 regs: ");
        for (int i = 0; i < 8; i++) { const int id = got[8 + i]; printf("(%d,%d)%s", id / dim_w, id % dim_w, i == 7 ? "\n" : " "); }
        // Is a lane's payload 8 CONSECUTIVE b16? That is the arrangement Marlin's dequant reads directly.
        int consec = 0;
        for (int l = 0; l < 32; l++) {
            bool c = true;
            for (int i = 1; i < 8; i++) if (got[l * 8 + i] != got[l * 8] + i) c = false;
            consec += c;
        }
        printf("    lanes whose 8 halves are contiguous in the source: %d/32 %s\n", consec,
               consec == 32 ? "(swzl already matches Marlin's packing)"
                            : "(so B must be pre-permuted -- which is what stage 2 tests)");
    }

    // ---- STAGE 2: invert the map. Build a source in which lane L's 8 halves ARE Marlin's 4 int32. ----
    if (!bijective_all) {
        printf("\n  STAGE 2 skipped: without a bijection there is no permutation to build.\n");
        printf("  => AIU/swzl cannot carry Marlin-format INT4; the Q4_K MoE stays on the Marlin derivation.\n");
        return 1;
    }
    probe<<<1, 32, shmem>>>(dG, dOut, dim_h, dim_w, 0, 0);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned short> map(32 * 8);
    CK(cudaMemcpy(map.data(), dOut, map.size() * 2, cudaMemcpyDeviceToHost));

    // want[lane][i] = the i-th b16 of lane `lane`'s desired int4 -- Marlin numbers them 8*lane + i.
    // map[lane][i] says slot i of lane `lane` comes from source position map[lane][i]. So place the wanted
    // value THERE and the load delivers it.
    std::vector<unsigned short> perm((size_t) dim_h * dim_w, 0xffff);
    for (int l = 0; l < 32; l++)
        for (int i = 0; i < 8; i++) perm[map[l * 8 + i]] = (unsigned short) (8 * l + i);
    CK(cudaMemcpy(dG, perm.data(), perm.size() * 2, cudaMemcpyHostToDevice));
    probe<<<1, 32, shmem>>>(dG, dOut, dim_h, dim_w, 0, 0);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned short> got2(32 * 8);
    CK(cudaMemcpy(got2.data(), dOut, got2.size() * 2, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int l = 0; l < 32; l++)
        for (int i = 0; i < 8; i++)
            if (got2[l * 8 + i] != (unsigned short) (8 * l + i)) bad++;
    printf("\n  STAGE 2 (pre-permuted source -> each lane's 4 int32, in order): %s (%d/%d wrong)\n",
           bad ? "FAIL" : "PASS", bad, 32 * 8);
    printf("  => %s\n", bad
        ? "the map is not invertible as assumed -- do NOT build the Q4_K MoE on AIU"
        : "AIU + swzl CAN deliver Marlin's per-lane int4 given a load-time repack we already do");
    return bad ? 1 : 0;
}
