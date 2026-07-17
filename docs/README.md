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