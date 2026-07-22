# docs

Detailed development and benchmark log for axon. The top-level `README.md`
stays deliberately lean (overview, roadmap, headline results) and links here
for the details.

Each devlog entry documents a completed investigation or optimization
iteration using the same structure: goal, setup, change, result (incl.
plot), interpretation, conclusion/next steps. Template:
[`devlog/TEMPLATE.md`](devlog/TEMPLATE.md).

## Index

| # | Period | Topic | Summary |
|---|--------|-------|---------|
| [001](devlog/001-mnist-warmup-calibration.md) | 2026-07-09 – 2026-07-15 | MNIST benchmark: warmup calibration | Established a reliable baseline measurement methodology (accounting for CPU thermal warm-up) before starting code optimization. |
| [002](devlog/002-matmul-at-hotspot.md) | 2026-07-18 | Profiling: identifying the matmul/at() hotspot | Sampling-profiler evidence that ~76% of runtime is spent in `Tensor::at()`, called from `matmul`'s naive triple loop. |
| [003](devlog/003-matmul-raw-stride-indexing.md) | 2026-07-19 | matmul: eliminate at() overhead via raw strided indexing | Replaced per-element `at()` calls with hoisted-stride pointer arithmetic; ~3.46x training throughput speedup. |
| [004](devlog/004-elementwise-odometer-iteration.md) | 2026-07-20 | elementwise binary ops: strided odometer iteration | Replaced per-element `at()`+heap-allocation in `+`/`-`/`*`/`/` with an incremental multi-index walk; ~2.09x further speedup (~7.22x cumulative). |
| [005](devlog/005-matmul-loop-order-dispatch.md) | 2026-07-22 | matmul: dispatch loop order on rhs contiguity | Two loop orders dispatched on `rhs` contiguity instead of one fixed order; ~4.04x further speedup (~29.17x cumulative). |