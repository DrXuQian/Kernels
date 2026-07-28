#include "cute/tensor.hpp"
#include "xplane_offline.hpp"
#include <cstdio>
int main() {
  auto x = xplane::tile_map_int1<64,64,128,32,32,2>();   // composed from plane 1
  auto o = xplane::plane_map<1,64,64,128,32,32,2>();     // plane 2's own single-plane rule
  long d=0, u=0; for (size_t i=0;i<x.size();++i){ if(x[i]<0)++u; else if(x[i]!=o[i])++d; }
  printf("composed vs own: %ld / %zu differ, %ld unset\n", d, x.size(), u);
  auto x1 = xplane::tile_map_int1<64,64,256,32,32,1>();
  auto o1 = xplane::plane_map<1,64,64,256,32,32,1>();
  long d1=0,u1=0; for (size_t i=0;i<x1.size();++i){ if(x1[i]<0)++u1; else if(x1[i]!=o1[i])++d1; }
  printf("GATE F2=1      : %ld / %zu differ, %ld unset  %s\n", d1, x1.size(), u1, (!d1&&!u1)?"OK":"FAIL");
  return 0;
}
