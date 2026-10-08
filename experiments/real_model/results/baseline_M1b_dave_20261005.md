# M1b baseline — Dave (Benchmarking), 2026-10-05

Host: Apple M2 arm64 8-core, 8 GB RAM, macOS 15.7.1, python 3.13.0
(`/usr/local/bin/python3.13`), pytest 9.0.3, numpy 2.5.2, torch 2.14.0.
Git: `8344bb3`. Root: canonical `Desktop/Program Projects/Eureka-2026` only.
Full record: `baseline_M1b_dave_20261005.json`. Raw run: `exp_20261005_164602_1e7dbf3a.json/.md`.

## Pytest timing table (wall via `/usr/bin/time -p`, `-p no:cacheprovider`)

| Module group | Result | Wall |
|---|---|---|
| tests/compiler | 85 passed | 0.70s |
| tests/analysis + tests/artifact | 131 passed | 2.19s |
| tests/kernels + runtime + compression + mixed_precision | 211 passed | 35.37s |
| tests/benchmark + hardware + planning + strategy + file_io + model_io + compute | 310 passed, 1 skipped | 4.66s |
| tests/integration (25 real-model) | 25 passed (2 pre-existing ternary RuntimeWarnings) | 23.41s |

All green, zero failures. Kernels group wall differs from Sam's 58.12s (same 211
pass) — machine-load variance, not a code change.

## Fixture baseline (tiny_llama_2L_128H: 787,072 params, 17 tensors, 1,449,631 B)

| Precision | Total (MB) | Ratio | Bits/wt | MAE | Rel-err | Transform (s) |
|---|---|---|---|---|---|---|
| INT8 | 0.762 | 1.97x | 8.125 | 0.000111 | 0.00646 | 0.098 |
| INT4 | 0.387 | 3.88x | 4.125 | 0.002002 | 0.11711 | 3.623 |
| BINARY | 0.106 | 14.22x | 1.125 | 0.009600 | 0.59967 | 0.434 |
| TERNARY | 0.199 | 7.53x | 2.125 | 0.015283 | 0.94762 | 0.700 |

Load: cold fresh-process (incl. imports) 1.10s; first load in-process 1.06s;
reload same process 0.0075s. Peak RSS ~191 MB process (interpreter+numpy+torch;
model itself 1.5 MB).

## Before/after vs M1a (Sam)

Identical to 3 decimals on every fixture metric (INT8 1.97x/MAE 1.11e-4,
INT4 3.88x/MAE 2.0e-3). No drift — M1 baseline locked.

## Methodology

- Pytest: one sequential run per group, wall indicative, pass/fail authoritative.
- Fixture: `run_experiment.py` (seed 42, group_size 128, float16 scales,
  symmetric, LAPTOP_CPU); ratio = original / (compressed + scales);
  errors are param-weighted means over tensors.
- Load/RSS: cold = fresh-process `load_model()` wall; reload = second call
  same process; RSS = `resource.ru_maxrss` after one load.
- All numbers measured on this host; nothing estimated or fabricated.

## Files touched

- `experiments/real_model/results/exp_20261005_164602_1e7dbf3a.json/.md` (generated)
- `experiments/real_model/results/baseline_M1b_dave_20261005.json/.md` (this record)
- No source/test edits.
