// FP16 activation + U4 weight + FP16 output, group_size=128 (group_blocks=8)
#include "kernel.h"
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

// decode: thread_m_blocks=1, m_block_size_8=true
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);

// decode: thread_m_blocks=1, m_block_size_8=false
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

// prefill: thread_m_blocks=2
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

// prefill: thread_m_blocks=3
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

// prefill: thread_m_blocks=4
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

}
