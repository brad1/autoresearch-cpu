# Plan: Replace GPU-only support with first-class CPU support

This plan translates the README request into a concrete implementation roadmap for making this repository run on CPU-only machines while keeping the existing 5-minute experiment workflow and agent loop intact.

## Goals

- Remove hard dependency on a single NVIDIA GPU for core workflows (`prepare.py`, `train.py`, autonomous loop).
- Keep the codebase small and readable (consistent with project philosophy).
- Preserve fair experiment accounting via fixed wall-clock budget, while adapting defaults to CPU throughput.
- Maintain backward compatibility for GPU users.

## Non-goals

- Implement distributed CPU training.
- Achieve GPU-level throughput on CPU.
- Add broad multi-backend complexity (full MPS/ROCm feature parity in this phase).

## Success criteria

- Fresh CPU-only environment can run:
  1. `uv sync`
  2. `uv run prepare.py`
  3. `uv run train.py`
- A 5-minute run completes without CUDA-specific crashes.
- Validation metric (`val_bpb`) logs normally.
- README and agent instructions document CPU mode and recommended tiny defaults.

---

## Phase 1 — Audit and design

1. **Identify GPU assumptions in code paths**
   - Device selection logic.
   - CUDA-only APIs (e.g., AMP/autocast settings, fused kernels, pinned memory assumptions, `torch.cuda.*` calls).
   - Attention/optimizer implementations that require CUDA kernels.
2. **Define minimal backend abstraction**
   - Single `device` selector with explicit priority: user override > auto-detect.
   - CPU-safe feature flags (e.g., disable fused/fp16-only paths).
3. **Decide CPU baseline config**
   - Conservative defaults for model depth, batch size, sequence length-compatible settings (if needed in `prepare.py`), eval tokens.

Deliverable: short design notes in code comments and/or `program.md` guidance.

## Phase 2 — Core runtime changes

1. **Device autodetection + override**
   - Add CLI/env override (e.g., `DEVICE=cpu` or `--device cpu`).
   - Fallback chain: CUDA -> CPU.
2. **Precision and autocast compatibility**
   - Ensure CPU path uses supported dtype (likely fp32; optional bf16 if supported).
   - Gate mixed precision and scaler logic by backend capability.
3. **Kernel/attention fallbacks**
   - Replace CUDA-only attention/ops with portable PyTorch implementation when on CPU.
   - Keep fast path for CUDA if available.
4. **Optimizer compatibility**
   - Ensure optimizer options that assume CUDA/fused ops are conditionally disabled on CPU.
5. **Data loading safety**
   - Remove/guard pinned-memory and GPU-transfer assumptions to avoid CPU overhead/errors.

Deliverable: `train.py` runs end-to-end on CPU and CUDA.

## Phase 3 — CPU-oriented defaults and ergonomics

1. **CPU profile presets**
   - Add a compact preset for CPU experiments (smaller depth/batch/eval tokens).
   - Keep current defaults for GPU unless CPU selected.
2. **Time-budget behavior on slow hardware**
   - Ensure 5-minute budget logic remains wall-clock based and robust when step time is large.
3. **Logging clarity**
   - Log backend, dtype, and active fallbacks at startup.

Deliverable: predictable CPU run behavior and understandable startup diagnostics.

## Phase 4 — Documentation and agent workflow updates

1. **README updates**
   - Replace GPU-only requirement language with multi-platform section including CPU support.
   - Add quick-start snippets for CPU mode.
   - Add expectations section (throughput tradeoffs and recommended small settings).
2. **`program.md` updates**
   - Instruct agents how to detect backend and select CPU-safe hyperparameter ranges.
   - Add guardrails to avoid proposing CUDA-only edits when running on CPU.

Deliverable: docs reflect reality and support autonomous CPU experimentation.

## Phase 5 — Validation matrix

Run and record:

1. **CPU smoke test**
   - `uv run prepare.py`
   - `uv run train.py` (5-minute budget)
2. **CUDA regression smoke test (if GPU available)**
   - Ensure no performance/feature regression in default GPU path.
3. **Determinism/basic sanity checks**
   - Verify loss decreases initially and `val_bpb` computes.

Deliverable: brief validation log in PR description.

---

## Risks and mitigations

- **Risk:** CPU runs are too slow to yield useful signal in 5 minutes.  
  **Mitigation:** introduce CPU preset with much smaller model/eval workload.

- **Risk:** Backend conditionals clutter single-file training code.  
  **Mitigation:** isolate backend checks near startup; expose simple booleans for downstream branches.

- **Risk:** Silent numerical differences across backends.  
  **Mitigation:** log dtype/backend explicitly and keep CPU default at fp32 first.

## Suggested implementation order (small PRs)

1. PR1: device detection + backend-safe precision/optimizer guards.
2. PR2: CPU fallback for attention/kernels and data movement paths.
3. PR3: CPU preset defaults + logging improvements.
4. PR4: README + `program.md` documentation refresh.

This sequence keeps each PR reviewable and minimizes breakage risk.
