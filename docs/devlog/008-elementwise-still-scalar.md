# 008 – re-profiling after matmul fixes: elementwise ops are still scalar

**Period:** 2026-07-26
**Commit(s):** none (investigation only, no code change)

## Goal

Per [007](007-matmul-cache-blocking-evaluation.md)'s conclusion: re-profile
with `mnist_profile` before deciding between a hand-written NEON kernel for
`matmul` and moving on to multi-threading, since the cost distribution from
[002](002-matmul-at-hotspot.md)/[006](006-matmul-autovectorization-check.md)
predates four rounds of changes (003–007) and shouldn't be assumed to still
hold.

## Setup

Same profiling setup as [002](002-matmul-at-hotspot.md)/[006](006-matmul-autovectorization-check.md):
`mnist_profile` built in Release, Xcode Instruments' Time Profiler, inverted
call tree.

## Result

Total runtime: 13.91 s (down from 56.78 s pre-[005](005-matmul-loop-order-dispatch.md),
consistent with that iteration's measured ~4x speedup). New self-time
breakdown:

| Function | Self-time |
|---|---|
| `operator+` | 17.7 % |
| `matmul` | 27.3 % (14.1 % via backward lambda, 13.2 % via `Linear::forward`) |
| `operator-` | 15.5 % |
| `operator*` | 14.6 % |
| `Tensor::sum` | 12.4 % |

`operator+`/`-`/`*` combined (47.8 %) now exceed `matmul` (27.3 %) — not
because they got slower (their combined ~6.65 s absolute cost is
statistically unchanged from the ~6.49 s measured in
[004](004-elementwise-odometer-iteration.md)'s follow-up profile), but
because `matmul`'s cost dropped so much (46.91 s → 3.80 s) that the
unchanged elementwise cost now dominates proportionally — the same
denominator-shrinks-so-the-rest-looks-bigger effect seen between
[003](003-matmul-raw-stride-indexing.md) and
[004](004-elementwise-odometer-iteration.md).

Followed the same auto-vectorization check as
[006](006-matmul-autovectorization-check.md): `-Rpass=loop-vectorize`
reports `elementwise_binary`'s outer per-element loop (`src/tensor.cpp:60`)
as "vectorized loop (vectorization width: 4, interleaved count: 4)". Assembly
inspection tells a different story — the hot loop compiles to exactly one
scalar operation per element:

```asm
LBB66_30:                        ; outer loop, one element
    ldr  s0, [x21, x9, lsl #2]   ; a_data[offset_a] — scalar register
    ldr  s1, [x22, x10, lsl #2]  ; b_data[offset_b] — scalar register
    fadd s0, s0, s1              ; one scalar add
    str  s0, [x25, x8, lsl #2]
LBB66_31:                        ; inner loop, the odometer carry
    ... integer bookkeeping (idx[]++, offset += stride, carry check) ...
```

No `.4s`/`.2s` vector registers, no `fmla`, anywhere in this function's hot
path. The vectorization remark almost certainly refers to something else in
the same source region — most likely the small `std::accumulate` call that
computes `num_elements` from `shape` — misattributed to a nearby line, not
to the actual per-element data loop.

## Interpretation

The odometer's data-dependent carry branch (`if (idx[d] < shape[d]) break;`,
`src/tensor.cpp:68`) blocks auto-vectorization of the outer loop: the
compiler cannot compute element `i+1`'s offsets without first resolving
whether element `i`'s increment carried into a higher dimension, so it can't
process four elements' worth of loads/stores at once.

This means [004](004-elementwise-odometer-iteration.md) traded one class of
overhead (`at()` + a per-element heap allocation) for a loop that stays
permanently scalar — a real, large improvement at the time, but incomplete
compared to what happened to `matmul`. After
[005](005-matmul-loop-order-dispatch.md), `matmul` benefits from *both*
overhead elimination *and* auto-vectorization; `elementwise_binary` only
ever got the former. The gap between them today is largely explained by
that difference, not by any remaining algorithmic overhead in the
elementwise path.

A second, smaller lesson: trusting a vectorization remark without checking
assembly would have been actively wrong here, not just insufficiently
confirmed — exactly the discipline [006](006-matmul-autovectorization-check.md)
established, now shown to matter in practice rather than just in principle.

## Conclusion / next steps

The next concrete target is `elementwise_binary`, not further work on
`matmul`: add a fast path for the common case (matching shapes, no
broadcasting, both operands contiguous) as a flat, branch-free loop —
mirroring the pattern `log()`/`exp()`/unary `operator-()` already use
(`is_contiguous()` check + `std::ranges::transform`) — falling back to the
general odometer only when broadcasting actually occurs. This is the same
"dispatch on the common case" strategy that worked for `matmul` in
[005](005-matmul-loop-order-dispatch.md). `Tensor::sum` (12.4 %) is a
secondary candidate worth a similar look afterward. Benchmark and re-profile
as usual once implemented.
