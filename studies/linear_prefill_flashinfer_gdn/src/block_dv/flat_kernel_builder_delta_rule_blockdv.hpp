/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "block_dv/flat_collective_tma_warpspecialized_delta_rule_blockdv.hpp"
#include "flashinfer/flat/hopper/kernel/flat_kernel_tma_warpspecialized_delta_rule.hpp"
#include "flashinfer/flat/hopper/kernel/flat_options.hpp"
#include "block_dv/flat_tile_scheduler_blockdv.hpp"
#include "flashinfer/flat/type_traits.hpp"

namespace flat::kernel {

template <class Element_, class ElementAccumulatorQK_, class ElementAccumulatorPV_,
          class TileShape_, int BlockDV_,  // BlkSeqQO, BlkSeqKV, HeadSizeQK + V slice
          class LayoutQ_, class LayoutK_, class LayoutV_, class LayoutO_, class DispatchPolicy,
          class Options = DefaultOptions>
struct FlatBuilderDeltaRule;

template <class Element, class ElementAccumulatorQK, class ElementAccumulatorPV,
          class TileShape, int BlockDV,  // BlkSeqQO, BlkSeqKV, HeadSizeQK + V slice
          class LayoutQ, class LayoutK, class LayoutV, class LayoutO, class Options>
struct FlatBuilderDeltaRule<Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShape, BlockDV, LayoutQ,
                            LayoutK, LayoutV, LayoutO,
                            cutlass::gemm::KernelTmaWarpSpecializedCooperative, Options> {
  using CollectiveMainloop = flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShape, BlockDV, LayoutQ, LayoutK, LayoutV,
      LayoutO, Options>;

  static constexpr bool kIsPersistent =
      find_option_t<Tag::kIsPersistent, false_type, Options>::value;
  static_assert(!kIsPersistent, "not implemented");

  static constexpr bool kIsGVA = find_option_t<Tag::kIsGVA, false_type, Options>::value;
  using GroupingTag = std::conditional_t<kIsGVA, GVATag, GQATag>;
  using TileScheduler = flat::kernel::IndividualTileScheduler<GroupingTag, BlockDV>;
  // using TileScheduler = std::conditional_t<kIsPersistent, flat::kernel::PersistentTileScheduler,
  // flat::kernel::IndividualTileScheduler>;

  using Kernel = flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<CollectiveMainloop,
                                                                     TileScheduler, Options>;
};

}  // namespace flat::kernel
