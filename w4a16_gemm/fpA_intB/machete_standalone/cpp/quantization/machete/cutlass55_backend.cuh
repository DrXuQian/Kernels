#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace machete_standalone {

struct Cutlass55Plan;

struct Cutlass55Config
{
    char const* name;
    int tile_n;
    int tile_m;
    int cluster_n;
    int cluster_m;
    int cluster_k;
};

std::vector<Cutlass55Config> supported_cutlass55_configs();
Cutlass55Config default_cutlass55_config();

void cutlass55_reorder_B_fp16_s4(cudaStream_t stream, cutlass::int4b_t const* b_in, cutlass::int4b_t* b_out,
    int k, int n);
void cutlass55_reorder_B_bf16_s4(cudaStream_t stream, cutlass::int4b_t const* b_in, cutlass::int4b_t* b_out,
    int k, int n);

void cutlass55_mm_fp16_s4(cudaStream_t stream, cutlass::half_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D, int m, int n, int k,
    int group_size, void* workspace, size_t workspace_bytes);
void cutlass55_mm_bf16_s4(cudaStream_t stream, cutlass::bfloat16_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, void* workspace, size_t workspace_bytes);

size_t cutlass55_get_workspace_size_fp16_s4(cutlass::half_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D, int m, int n, int k,
    int group_size, Cutlass55Config config);
size_t cutlass55_get_workspace_size_bf16_s4(cutlass::bfloat16_t const* A, cutlass::int4b_t const* B_reordered,
    cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C, cutlass::bfloat16_t* D, int m, int n,
    int k, int group_size, Cutlass55Config config);

Cutlass55Plan* cutlass55_create_plan_fp16_s4(cudaStream_t stream, cutlass::half_t const* A,
    cutlass::int4b_t const* B_reordered, cutlass::half_t const* scales, cutlass::half_t const* C, cutlass::half_t* D,
    int m, int n, int k, int group_size, Cutlass55Config config, void* workspace, size_t workspace_bytes);
Cutlass55Plan* cutlass55_create_plan_bf16_s4(cudaStream_t stream, cutlass::bfloat16_t const* A,
    cutlass::int4b_t const* B_reordered, cutlass::bfloat16_t const* scales, cutlass::bfloat16_t const* C,
    cutlass::bfloat16_t* D, int m, int n, int k, int group_size, Cutlass55Config config, void* workspace,
    size_t workspace_bytes);
void cutlass55_run_plan(Cutlass55Plan* plan, cudaStream_t stream);
void cutlass55_destroy_plan(Cutlass55Plan* plan);

} // namespace machete_standalone
