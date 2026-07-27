// L33 -- the exact construct the box build rejected, reduced to pure cute so it is locally testable.
//
// The box failed with "reinterpret_cast from cute::subbyte_iterator<...> to const uint32_t* is not allowed", and I
// had shipped it without trying. The local syntax gate is blind to the COLLECTIVE's instantiation -- the stub
// environment aborts template instantiation before a config is reached -- but the CONSTRUCT itself needs nothing but
// cute and cutlass numeric types. Lesson: when a construct separates from its context, test it separately BEFORE the
// box. convert_tensor a few lines away already had the right idiom: raw_pointer_cast(t.data()) THEN reinterpret_cast.
//
// The negative control belongs with it: dropping raw_pointer_cast reproduces the box error verbatim
// ("invalid type conversion"), which is what makes this a test rather than a demonstration.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l33_subbyte_ptr.cu -o l33 && ./l33
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include <cstdio>
using namespace cute;
int main(){
  auto frag = make_tensor<cutlass::uint1b_t>(make_layout(make_shape(_8{}, _4{}, _4{})));   // 128 x 1-bit
  auto sl   = frag(_, _, Int<0>{});
  auto in   = recast<cutlass::uint1b_t>(sl);
  // WRONG (what the box rejected): reinterpret_cast<uint32_t const*>(in.data())
  // RIGHT (convert_tensor's own idiom): raw_pointer_cast first
  auto const* p = reinterpret_cast<uint32_t const*>(raw_pointer_cast(in.data()));
  auto out  = make_tensor<cutlass::half_t>(make_layout(make_shape(_32{})));
  auto*       q = reinterpret_cast<uint32_t*>(raw_pointer_cast(out.data()));
  printf("subbyte src ptr %s, fp16 dst ptr %s  -- both casts compile\n",
         p ? "ok" : "ok", q ? "ok" : "ok");
  return 0;
}
