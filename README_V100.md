# triton-v100: Triton 3.8 with Volta and Turing tensor cores

Upstream Triton 3.x lowers `tl.dot` to FMA instructions on sm_70 (Volta) and on sm_75 (Turing)
only uses part of the tensor-core paths. This repository is Triton **v3.8.0** with the Volta /
Turing tensor-core work of [Peuqui/triton](https://github.com/Peuqui/triton) (branch
`sm7x-tensor-cores`, written against v3.6.0) ported on top:

- Volta: `tl.dot` on tensor cores through MMAv1 (`mma.sync.m8n8k4`), expressed with linear layouts;
  operand row/col chosen per operand from its memory order; an 8x32 tile for dots with at most 8
  rows (decode); batched dots; a 16-row shared-memory swizzle.
- Turing: MMAv2 (`m16n8k8`) re-enabled, batched dots, narrow operands kept narrow in shared memory.
- Before sm_80 (no `cp.async`): software pipelining through registers with double-buffered shared
  memory, one barrier per main-loop iteration on Volta, small loop-invariant operands kept in
  registers by register budget.
- Fixes: FMA dot with fp16/bf16 accumulators sums in fp32; int8 FMA operands widen as integers;
  reductions whose lane bits along the axis are scattered.

Why v3.8.0 and not `main`: vLLM 0.33 runs on torch 2.14, which pins `triton==3.8.0`. The default
branch `sm7x-on-3.8` is therefore v3.8.0 (the 3.8 release branch left `main` on 2026-06-23) plus
the 26 ported commits; GitHub's "commits behind main" count is the unreleased `main` work since
then, deliberately not included.

Each ported commit carries `(cherry picked from commit ...)` pointing at its original on
`sm7x-tensor-cores`. Where 3.8 had changed the same code, the port keeps the 3.8 behaviour and adds
the Volta / Turing path next to it (Prefetch keeps 3.8's local_load token tracking; the NVIDIA
barrier filter keeps 3.8's `canSkipBarSync` and adds the register-staged filter below sm_80).

Used by [ARC-HXT/vllm-v100](https://github.com/ARC-HXT/vllm-v100).

## Build

Same as upstream Triton (`pip install -e .` or `pip wheel . --no-build-isolation`); the build
downloads the pinned LLVM and NVIDIA tools. Wheels for CPython 3.13 / x86_64 are built by the
`V100 wheel` workflow in this repository.

## Tests

Test results on V100 and the comparison with stock Triton 3.8.0 are recorded in the release notes.
