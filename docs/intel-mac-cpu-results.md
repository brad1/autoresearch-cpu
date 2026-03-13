# Intel Mac CPU Results

This page summarizes the local CPU-only experiment sweep run on one Intel Mac after the compatibility and locked-install work landed in this fork.

## Comparison Rules

- All runs use the same local CPU comparison setup: `TIME_BUDGET=300` and `CPU_EVAL_TOKENS=524288`.
- These numbers are directly comparable within this machine/profile only.
- They are not 1:1 with the default upstream H100-style setup because the CPU path uses a reduced eval budget and a smaller local runtime profile.

## Run Summary

| Commit | Change | val_bpb | training_seconds | total_seconds | num_steps | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `4de41db` | baseline Intel-Mac CPU compatibility patch | 2.693582 | 308.1 | 524.2 | 22 | superseded |
| `a7f18d2` | halve CPU total batch size to 4096 | 2.601130 | 302.4 | 426.1 | 70 | superseded |
| `0e59b77` | reduce CPU depth to 3 with 4096 total batch | 2.405416 | 300.0 | 405.7 | 134 | current best |
| `d9d97fe` | reduce `HEAD_DIM` to 64 | 2.492597 | 301.8 | 435.4 | 95 | discard |
| `65ff0fe` | reduce CPU depth to 2 | 2.497718 | 303.5 | 639.5 | 106 | discard |
| `f3bf880` | halve CPU batch to 2048 | 2.412210 | 300.5 | 433.9 | 189 | discard |
| `a623f57` | shorten warmdown ratio to 25% | 2.465286 | 302.7 | 448.8 | 99 | discard |

## Notes

- The best local profile is still commit `0e59b77`: `CPU_DEPTH=3`, `CPU_DEVICE_BATCH_SIZE=2`, `CPU_TOTAL_BATCH_SIZE=2**12`, `CPU_EVAL_BATCH_SIZE=16`, `CPU_EVAL_TOKENS=524288`, `CPU_WINDOW_PATTERN="L"`.
- The strongest follow-up negative result was `f3bf880`, which increased training steps to `189` but still regressed slightly on `val_bpb`.
- Local run logs are archived under `results/intel-mac-2026-03-13/` and intentionally ignored by git.
