# 006 – checking whether matmul is already auto-vectorized

**Period:** 2026-07-23
**Commit(s):** none (investigation only, no code change)

## Goal

Per the plan after [005](005-matmul-loop-order-dispatch.md): before attempting
cache blocking or a hand-written NEON kernel for `matmul`, check whether
`-O3` already auto-vectorizes the current scalar loops, and whether adding
`__restrict` to `lhs_data`/`rhs_data`/`result`'s buffer pointer unlocks
vectorization the compiler can't otherwise prove is safe. If the compiler
is already emitting good SIMD code, hand-written SIMD has a much smaller
ceiling than expected — worth knowing before investing the effort.

## Setup

The existing `build/` directory is configured `Debug` (`-O0` +
sanitizers, see `CMakeLists.txt:46`) — not representative of what actually
runs in benchmarks. A separate `build-release/` directory was configured
instead (`cmake -B build-release -DCMAKE_BUILD_TYPE=Release
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, gitignored like all `build-*/` dirs) to
extract the exact flags used for the real Release build
(`-O3 -DNDEBUG -std=gnu++20 -arch arm64`, from `compile_commands.json`).

`tensor.cpp` was then recompiled directly with those flags plus Clang's
vectorization-remark flags:

```
-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
```

and separately with `-S` to inspect the emitted assembly directly.

## Result

Both inner loops in `Tensor::matmul` (`src/tensor.cpp:370`) — the i-k-j fast
path (`src/tensor.cpp:400`) introduced in
[005](005-matmul-loop-order-dispatch.md), and the i-j-k fallback
(`src/tensor.cpp:408`) used for the grad-input case — already report:

```
remark: vectorized loop (vectorization width: 4, interleaved count: 4) [-Rpass=loop-vectorize]
```

Confirmed in the generated assembly: real NEON fused-multiply-add
instructions (`fmla.4s`, operating on 4 lanes of 32-bit floats each) are
emitted for the accumulation, not just scalar code.

No "run-time check" remark appears for either loop anywhere in the report —
unlike other vectorized loops elsewhere that require a generated runtime
aliasing check between pointers before the vectorized path can safely run.
The compiler evidently proved on its own that `result`'s freshly-allocated
buffer (`result = zeros(...)`, visibly a new allocation) cannot alias
`lhs_data`/`rhs_data`, without needing `__restrict` as a hint.

## Interpretation

`matmul`'s scalar loops were already compiled down to width-4, 4×-interleaved
NEON code with no aliasing overhead, before any manual SIMD work. This means:

- `__restrict` was tried as a hypothesis but isn't expected to change
  anything here — there's no aliasing-check cost to remove, and the
  vectorization width is already at the natural limit for `float32x4_t`
  (4 lanes) on this architecture.
- The ~29× cumulative speedup through [005](005-matmul-loop-order-dispatch.md)
  came entirely from eliminating overhead (`at()` calls, per-element
  allocations) and improving memory access order — not from unlocking
  vectorization that wasn't happening. The compiler was vectorizing the
  arithmetic all along; it just had comparatively little arithmetic-only
  time to work with once the surrounding overhead was this large.

## Conclusion / next steps

A hand-written NEON kernel would not be "vectorizing unvectorized code" —
it would be competing with an already
decent auto-vectorized baseline (width 4, interleave 4, no aliasing
overhead). That's a smaller ceiling than the last three iterations, but
still a meaningful one: hand-written kernels can typically improve on
auto-vectorized code through better register blocking/tiling (reusing
loaded values across more output elements per load, reducing the ratio of
loads to FMAs) and explicit multi-level cache blocking, which the
auto-vectorizer does not attempt on its own. Next step per the original
plan: a single-level cache-blocking pass over `matmul`'s `i`/`inner`/`c`
loops, sized empirically against the M2's cache hierarchy, measured with
`mnist_benchmark` before committing to writing NEON intrinsics by hand.