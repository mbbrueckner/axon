# axon

A neural network library from scratch in C++.

## roadmap

- [X] Tensor class
- [X] Autograd
- [X] basic Modules
- [X] Optimzers (SGD)
- [X] first test - model (XOR)
- [X] first real model MNIST
- [ ] performance benchmarks
- [ ] Code optimization
- [ ] Code parallelization
- [ ] addition of more modules

## XOR (`examples/xor_demo.cpp`) results

**Setup:** A 2-4-1 MLP (`Linear(2,4) → ReLU → Linear(4,1)`) trained on all four
XOR samples (full-batch) for 1000 epochs with SGD (learning rate 0.1) against
the mean squared error.

```
./build/examples/xor
Epoch: 0; loss:0.522413; acc.:0.5
Epoch: 100; loss:0.0178309; acc.:1
Epoch: 200; loss:0.000173242; acc.:1
Epoch: 300; loss:1.57907e-06; acc.:1
Epoch: 400; loss:1.41426e-08; acc.:1
Epoch: 500; loss:1.26356e-10; acc.:1
Epoch: 600; loss:1.56431e-12; acc.:1
Epoch: 700; loss:6.14889e-13; acc.:1
Epoch: 800; loss:4.01639e-13; acc.:1
Epoch: 900; loss:3.23491e-13; acc.:1
```

**Interpretation:**

- The model reaches **100 % accuracy by epoch 100** and drives the loss down by
  roughly **twelve orders of magnitude**, flattening near `~1e-13` — the
  effective precision floor of 32-bit floats.
- XOR is **not linearly separable**, so fitting it requires the hidden ReLU
  layer. Successfully learning it end-to-end validates that the full pipeline —
  tensor ops, autograd/backprop, and the SGD update — works correctly.

**Caveats:**

- Weights are initialized from `std::random_device` (see `Linear`), so the exact
  loss values vary between runs; the convergence behaviour does not.
- This is a **correctness check, not a generalization benchmark**: with only four
  samples and no train/test split, the network simply memorizes the truth table.
  Measuring real generalization is what the MNIST roadmap item is for.

## MNIST - Performance benchmark

**Setup:**

Sequetial model: Linear(784, 128) -> ReLU -> Linear (128,10)
Optimizer: SGD

| Metric                                    | Value                     |
|-------------------------------------------|---------------------------|
| Parameters                                | 101,770                   |
| Training images                           | 60,000 (28×28)            |
| Batch size                                | 32 → 1,875 batches/epoch  |
| FLOPs per sample (fwd+bwd)                | ≈ 0.61 MFLOP              |
| FLOPs per epoch                           | ≈ 36.6 GFLOP              |
| FLOPs per run (5 epochs)                  | ≈ 183 GFLOP               |
| Total FLOPs (5 runs + warmup)             | ≈ 952 GFLOP (~0.95 TFLOP) |
| Throughput (unoptimized, single-threaded) | ≈ 13.5 MFLOP/s            |

*FLOPs estimated using the standard approximation for fully-connected
layers: forward ≈ 2·B·in·out (+ bias), backward ≈ 4·B·in·out
(grad_input + grad_weight, + bias gradient); ReLU/loss are negligible.*

**unoptimized baseline:**

<img src="benchmarks/mnist/plots/mnist_benchmark_3bb5b8e_2026-07-09T19:59:18.png" width="700"/>

The plot clearly shows that the current warm-up phase is insufficient: runs 0 and 1
are both noticeably slower (≈90 s and ≈84 s mean epoch time) and more variable
(larger stddev) than runs 2–9, which settle into a stable ≈81 s baseline.

Looking at the per-epoch trace, the anomaly is *not* a simple monotonic
cold-cache warm-up: epoch 0 of run 0 (≈82 s) is actually among the fastest
epochs measured, times then climb to ≈95 s over the rest of run 0, stay
elevated through the first two epochs of run 1, and only then drop sharply to
the stable ≈81 s baseline that holds for the remaining ~8 runs. That
fast → slow → stable shape points less at classic cache/allocator warm-up
and more at CPU power-state transients (e.g. an initial turbo-boost burst
followed by thermal throttling under sustained load, before the clock settles
into a steady state) — the anomaly spans roughly 13 epochs, i.e. ~19 minutes
of continuous load, which lines up with typical laptop thermal ramp-up time.

**Conclusion:** The single untimed warm-up epoch is nowhere near enough to
reach steady state, and a fixed "discard N runs" rule is a rough
approximation since the transition doesn't align cleanly with a run boundary.
For now, excluding runs 0 and 1 from the aggregated statistics is a
reasonable and simple correction to get a representative throughput number.
Longer term, the warm-up should be convergence-based (run until several
consecutive epochs land within a small tolerance of each other, e.g. ±2%)
rather than a fixed epoch count, and it would be worth logging CPU
frequency/temperature (e.g. via `powermetrics` on macOS) alongside the
timings to confirm the throttling hypothesis. Pinning the benchmark process
to performance cores or disabling frequency scaling would also make
before/after comparisons from future optimization work less noisy.

**adaptive Warmup:**

<img src="benchmarks/mnist/plots/mnist_benchmark_12094ac_2026-07-13T20:44:38Z.png" width="700"/>

The runtime graph of the first run with adaptive warmup points out well, that the warmup still didn't fix the problem.
It appears that the warm-up phase converged on a coincidental early plateau
rather than the true steady state: `warmup.csv` shows only 3 epochs
(≈81.4 s, ≈81.5 s, ≈81.5 s — a spread of ~0.13 %, well inside the ±2 %
tolerance) before the convergence check declared the system stable. The
measured runs that follow tell a different story: run 0 stays low for its
first three epochs (~81 s) and then climbs to a peak of ≈93.0 s at epoch 6,
before settling back down; run 1 is still elevated for its first three
epochs (~85.5–86.8 s) before dropping to the ~81–83 s baseline that holds
from run 2 onward. The thermal ramp-up simply happened to pause briefly
around the point where the warm-up loop was sampling, then resumed — a
short 3-epoch window landed inside that pause and mistook it for the end of
the transient.

**Conclusion:** A short convergence window is not robust against a
non-monotonic warm-up curve — it can be satisfied by chance during a
temporary lull, before the real thermal ramp has even started. Since the
empirical ramp-up in the earlier fixed-warmup measurement spanned ~13
epochs, the adaptive warm-up now enforces a `MIN_WARMUP_EPOCHS` floor (12)
below which convergence isn't checked at all, and the window was widened
from 3 to 5 epochs to make a coincidental plateau less likely to trigger a
false positive. This should be re-validated with another overnight run
before being trusted as the final warm-up strategy.