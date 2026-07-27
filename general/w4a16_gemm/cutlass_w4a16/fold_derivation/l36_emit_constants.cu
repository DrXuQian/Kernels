// L36 -- the emitter's lop3/fma constants are DERIVABLE from Bits, so int1 and int2 need one emitter, not two.
//
// MixGemmInt1Emit is int1-only by name and by its 16 hardcoded (mask, mul, add) triples; int2's converter has its own
// 8. That is two tables encoding one rule, and it means chunking int2 would mean copying the whole emitter. But the
// two tables are the SAME formula at different strides:
//
//     pairs per vreg   = 16/Bits            (int1 16, int2 8)
//     per level        = 8/Bits             (level = t / (8/Bits); level 1 reads reg>>8)
//     bit position b   = (t % (8/Bits)) * Bits
//     mask             = m | (m<<16),  m = ((1<<Bits)-1) << b
//     mul              = fp16(2^-b)        -> half pattern (15 - b) << 10
//     add              = fp16(-2^(10-b))   -> half pattern 0x8000 | ((25 - b) << 10)
//
// Checked below against the constants actually written in fast_numeric_conversion_for_mix_gemm.h for BOTH widths.
// int4 is deliberately out of scope: it uses the +8 bias with the 1032/72 magic on nibbles, a different scheme -- and
// it does not need chunking anyway, since slots/delivery = 4 leaves it one k-atom per copy step, i.e. NChunk = 1.
//
//   nvcc -std=c++17 -Istub_inc -I../../../../third_party/actlize/include l36_emit_constants.cu -o l36 && ./l36
#include <cstdio>
#include <cstdint>

static uint32_t dup16(uint32_t h) { return h | (h << 16); }
static uint32_t mask_of(int Bits, int b) { return dup16((uint32_t)(((1u << Bits) - 1u) << b)); }
static uint32_t mul_of (int b)           { return dup16((uint32_t)((15 - b) << 10)); }
static uint32_t add_of (int b)           { return dup16((uint32_t)(0x8000u | ((25 - b) << 10))); }

struct Row { int t; uint32_t mask, mul, add; int level; };

// transcribed from the converters as they stand
static const Row kInt1[16] = {
  {0,0x00010001,0x3c003c00,0xe400e400,0},{1,0x00020002,0x38003800,0xe000e000,0},
  {2,0x00040004,0x34003400,0xdc00dc00,0},{3,0x00080008,0x30003000,0xd800d800,0},
  {4,0x00100010,0x2c002c00,0xd400d400,0},{5,0x00200020,0x28002800,0xd000d000,0},
  {6,0x00400040,0x24002400,0xcc00cc00,0},{7,0x00800080,0x20002000,0xc800c800,0},
  {8,0x00010001,0x3c003c00,0xe400e400,1},{9,0x00020002,0x38003800,0xe000e000,1},
  {10,0x00040004,0x34003400,0xdc00dc00,1},{11,0x00080008,0x30003000,0xd800d800,1},
  {12,0x00100010,0x2c002c00,0xd400d400,1},{13,0x00200020,0x28002800,0xd000d000,1},
  {14,0x00400040,0x24002400,0xcc00cc00,1},{15,0x00800080,0x20002000,0xc800c800,1}};
static const Row kInt2[8] = {
  {0,0x00030003,0x3c003c00,0xe400e400,0},{1,0x000c000c,0x34003400,0xdc00dc00,0},
  {2,0x00300030,0x2c002c00,0xd400d400,0},{3,0x00c000c0,0x24002400,0xcc00cc00,0},
  {4,0x00030003,0x3c003c00,0xe400e400,1},{5,0x000c000c,0x34003400,0xdc00dc00,1},
  {6,0x00300030,0x2c002c00,0xd400d400,1},{7,0x00c000c0,0x24002400,0xcc00cc00,1}};

static bool check(int Bits, const Row* rows, int n) {
  const int per_level = 8 / Bits;
  long bad = 0;
  for (int i = 0; i < n; ++i) {
    const int t = rows[i].t, level = t / per_level, b = (t % per_level) * Bits;
    const uint32_t m = mask_of(Bits, b), u = mul_of(b), a = add_of(b);
    const bool ok = (m == rows[i].mask) && (u == rows[i].mul) && (a == rows[i].add) && (level == rows[i].level);
    if (!ok) { ++bad; if (bad <= 3)
      printf("    int%d t%2d: mask %08x/%08x  mul %08x/%08x  add %08x/%08x  level %d/%d\n",
             Bits, t, m, rows[i].mask, u, rows[i].mul, a, rows[i].add, level, rows[i].level); }
  }
  printf("  int%d: %d rows, %ld differ from the formula  %s\n", Bits, n, bad,
         bad ? "<-- NOT one rule" : "<-- derivable, one emitter suffices");
  return bad == 0;
}

int main() {
  printf("L36 -- are the emitter constants derivable from Bits?\n\n");
  bool ok = check(1, kInt1, 16) && check(2, kInt2, 8);
  printf("\n  pairs/vreg = 16/Bits, per level = 8/Bits, b = (t %% (8/Bits))*Bits\n");
  printf("  mask = m|(m<<16) with m = ((1<<Bits)-1)<<b ; mul = fp16(2^-b) = (15-b)<<10 ;\n");
  printf("  add = fp16(-2^(10-b)) = 0x8000 | ((25-b)<<10)\n");
  printf("\n  ALL: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
