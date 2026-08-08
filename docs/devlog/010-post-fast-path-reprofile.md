# 010 – re-profiling after the elementwise fast path

**Period:** 2026-08-06
**Commit(s):** none (investigation only, no code change)

## Goal

[009](009-elementwise-fast-path.md)'s two profiling attempts were both
contaminated (stale build, then likely background load) and gave no
trustworthy picture of where time goes now. Get a clean profile on an idle
machine before picking the next target, per that entry's conclusion.

## Setup

Same profiling setup as previous iterations: `mnist_profile` built in
`build-release` (confirmed `-O3`/`Release`), Xcode Instruments' Time
Profiler, inverted call tree. This run shows "Thermal State: Nominal" and a
tight, uninterrupted ~10 s CPU trace — no signs of the contention that
undermined both attempts in 009.

## Result

Total runtime: 9.61 s. Self-time breakdown:

| Function | Self-time |
|---|---|
| `matmul` | 42.1 % |
| `operator*` | 19.7 % |
| `Tensor::sum` | 16.1 % |
| `shared_ptr::allocate_shared` | 4.2 % |
| `operator+` | 3.3 % |
| `operator-` | ~2 % |

`operator+`/`operator-` have dropped to a few percent each — consistent
with [009](009-elementwise-fast-path.md)'s fast path working as intended.
`operator*`, by contrast, is still the second-largest single cost, and
`Tensor::sum` — untouched by any of the nine iterations so far — has
emerged as the third-largest.

Looking at `Tensor::sum`'s two overloads (`src/tensor.cpp:648` and `:676`):
the no-argument full reduction already has an `is_contiguous()` fast path
via `std::accumulate`. The per-dimension overload, `sum(idx_t dim, bool
keep_dim)`, does not — it calls `utils::flat_to_indices(i, shape_)` for
every element, allocating a fresh `std::vector` per element on the heap,
the same class of problem fixed for `matmul` in
[003](003-matmul-raw-stride-indexing.md) and for the elementwise operators
in [004](004-elementwise-odometer-iteration.md)/[009](009-elementwise-fast-path.md).
This overload is what `reduce_grad_to_shape` (`src/tensor.cpp:28`) calls
whenever a backward pass needs to undo broadcasting — e.g. the bias
gradient in `Linear::forward`'s `input.matmul(weights_) + bias_`, which runs
on every training step.

## Interpretation

Two independent findings, not one:

1. **`Tensor::sum(dim, keep_dim)` is a straightforward instance of the
   already-familiar bug** (per-element heap allocation via
   `flat_to_indices`, no direct stride arithmetic) that was simply never
   addressed because attention was on `matmul` and the binary operators.
   Same fix pattern should apply directly.
2. **`operator*` is a different kind of problem.** Unlike `+`/`-`, which
   are dominated by same-shape gradient-accumulation calls (`operator+=`
   in every backward step, per [004](004-elementwise-odometer-iteration.md)'s
   finding), a likely dominant caller of `operator*` is
   `SGD::step()`'s `lr * grad`, where `lr` becomes a `{1}`-shaped tensor
   multiplied against a full parameter gradient — an inherent broadcast,
   which can never take the fast path added in
   [009](009-elementwise-fast-path.md) regardless of how it's implemented.
   If that hypothesis holds, `operator*`'s remaining cost isn't fixable by
   the same technique; it would need either a dedicated scalar-broadcast
   fast path (no index/stride machinery needed at all when one operand is
   a true 1-element tensor) or a different approach to scalar multiplication
   entirely. This should be verified (e.g. by checking `operator*`'s callers
   in this profile) rather than assumed before deciding how to fix it.

## Conclusion / next steps

Two candidate targets, likely two separate iterations:

- **`Tensor::sum(dim, keep_dim)`**: apply the same branch-free/direct-stride
  approach used for `matmul` and the elementwise operators. Should be
  mechanical given the precedent.
- **`operator*`**: confirm the scalar-broadcast hypothesis first (check its
  callers in the call tree), then decide on a fix — likely a dedicated
  fast path for the "one operand is a true scalar" case, distinct from the
  contiguous-matching-shapes fast path already added.

Recommend starting with `sum(dim, keep_dim)` since the fix is already
well-understood, then re-profiling before committing to an approach for
`operator*`.
