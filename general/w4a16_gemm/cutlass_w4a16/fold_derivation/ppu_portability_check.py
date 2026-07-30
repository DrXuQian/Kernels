#!/usr/bin/env python3
"""Box-built sources must not name NVIDIA-only identifiers in code the box actually compiles.

WHY THIS EXISTS, AND WHY IT IS NOT A PLAIN GREP. syntax_check.sh reported 0 noise lines for these very files
and the box then failed on `fatal error: cuda_fp16.h: No such file or directory`. It could not have caught it:
local nvcc HAS cuda_fp16.h whether or not -D__HGGCCC__ is passed, so a wrong-platform INCLUDE parses fine and
only the box disagrees. The same blind spot covers cudaStream_t in a signature and __halves2half2, an
intrinsic that exists in CUDA and appears NOWHERE in actlize.

A plain grep is not enough either, because the portable spelling is itself written as
    #if defined(__HGGCCC__) ... #else <the CUDA one> #endif
so the CUDA name is PRESENT in every correctly-ported file. The check therefore evaluates the same conditions
the box does -- __HGGCCC__ defined, ENABLE_BF16 not -- and only reports hits in the live branches. That makes
it exact rather than heuristic, and it is the difference between a check that is trusted and one that gets a
growing exception list until it hides the next real hit.

Identifiers that exist in CUDA *and* in actlize are deliberately NOT on the list: half2, __hfma2, __hsub2,
__hadd2, __hmul2, __half2half2, __shfl_xor_sync, __syncthreads, uint4, uint2, dim3, float2, __float2half and
__half2float were all confirmed present in the actlize include tree, so they are portable.
"""
import os, re, sys

DENY = re.compile(r'cuda_fp16\.h|cuda_bf16\.h|cuda_runtime\.h|cudaStream_t|cudaMalloc|cudaFree|cudaMemcpy'
                  r'|cudaDeviceSynchronize|cudaGetLastError|cudaError_t|cudaSuccess|cudaEvent'
                  r'|__halves2half2|__halves2bfloat162|__nv_bfloat')
# gemv_rt.hpp's whole job is to straddle the two runtimes, so it names both by design.
ALLOW = {'gemv_rt.hpp'}
# What the box defines. Anything guarded on something else is assumed live (conservative: reports more).
DEFINED = {'__HGGCCC__'}
UNDEFINED = {'ENABLE_BF16'}

def live_lines(path):
    """Yield (lineno, text) for lines the box would compile, honouring #if/#else/#endif on the macros above."""
    stack = []           # each entry: True (live) / False (dead) / None (unknown -> treat as live)
    out = []
    cond = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')
    def truth(expr):
        e = expr.strip()
        for m in DEFINED:
            if f'defined({m})' in e or e == m: return True
        for m in UNDEFINED:
            if f'defined({m})' in e or e == m: return False
        return None
    for i, line in enumerate(open(path, errors='replace'), 1):
        m = cond.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ('if', 'ifdef'):
                stack.append(truth(rest.replace('defined', 'defined')) if kind == 'if' else truth(f'defined({rest.strip()})'))
            elif kind == 'ifndef':
                t = truth(f'defined({rest.strip()})')
                stack.append(None if t is None else (not t))
            elif kind == 'elif':
                if stack: stack[-1] = truth(rest)
            elif kind == 'else':
                if stack: stack[-1] = None if stack[-1] is None else (not stack[-1])
            elif kind == 'endif':
                if stack: stack.pop()
            continue
        if all(s is not False for s in stack):
            out.append((i, line.rstrip('\n')))
    return out

def main():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    files = [os.path.join(here, f) for f in sorted(os.listdir(here))
             if f.endswith(('.cu', '.cuh', '.hpp', '.h'))]
    sub = os.path.join(here, 'gemv_lowbit')
    if os.path.isdir(sub):
        files += [os.path.join(sub, f) for f in sorted(os.listdir(sub))]

    bad = 0
    for f in files:
        if os.path.basename(f) in ALLOW: continue
        for ln, text in live_lines(f):
            s = text.split('//')[0]
            if DENY.search(s):
                print(f"  [FAIL] ppu_portability: {os.path.relpath(f, here)}:{ln} is NVIDIA-only in a branch "
                      f"the box compiles:\n           {text.strip()}")
                bad = 1

    # TWO SELF-CHECKS, because a portability check that silently stops matching is worse than none.
    #
    # (1) The deny list must still match REAL code somewhere. gemv_rt.hpp is the file that names both runtimes,
    #     so it is the anchor -- but its cuda* names sit in the #else of #if defined(__HGGCCC__), i.e. in a DEAD
    #     branch, so the anchor has to be checked against the raw text and not against the live subset. Getting
    #     that backwards made this self-check fail on a correctly-ported tree the first time it ran.
    rt = os.path.join(sub, 'gemv_rt.hpp')
    if not (os.path.exists(rt) and DENY.search(open(rt, errors='replace').read())):
        print("  [FAIL] ppu_portability: the deny list matches nothing even in gemv_rt.hpp -- vacuous check")
        return 1
    # (2) An UNCONDITIONAL cuda include must be reported. This is the exact shape that reached the box.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, 'probe.cuh')
        open(probe, 'w').write("#include <cuda_fp16.h>\nint x;\n")
        if not any(DENY.search(t.split('//')[0]) for _, t in live_lines(probe)):
            print("  [FAIL] ppu_portability: an unconditional cuda_fp16.h include is NOT reported -- the "
                  "liveness filter is swallowing everything")
            return 1
    if not bad:
        print(f"  [ok]   ppu_portability: {len(files)} box-built sources are in the portable subset")
    return bad

sys.exit(main())
