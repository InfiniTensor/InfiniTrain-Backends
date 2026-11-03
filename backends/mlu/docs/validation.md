# MLU backend - validation status (stage snapshot)

This file records what has actually been run and what has not. It is deliberately conservative:
anything not listed as verified below should be treated as unverified.

## Verified on 2026-11-03 (this provider layout)

Built from a fresh CMake cache against the pinned InfiniTrain content (`c90338c`, v0.7.0) with
GCC 13.4.0, cncc 5.7.6, CMake 4.4.5 and Ninja 1.13.2; targets `mlu_gpt2` and `mlu_op_test`.

| Item | Evidence |
|---|---|
| Fresh configuration and build | `configure` and `build` of `mlu_gpt2` + `mlu_op_test` both succeeded; all **9** BANG C sources were compiled by `cncc` |
| Operator comparison vs CPU (`op_test`) | 35 small-shape FP32 cases: Linear (both weight layouts, bias on/off, non-square 3x5->7), LayerNorm, Softmax, Add broadcast, MulScalar, Transpose, CrossEntropy, Pow domain and Pow(x,3) fwd/bwd. All pass with `abs(got-ref) <= 1e-5 + 1e-4*abs(ref)` (Pow uses 1e-6/1e-5) |
| Negative comparison paths | `--inject-finite` and `--inject-nan` return exit code 1 with `RESULT: FAIL`; the controlled fatal case aborts with the expected marker and no post-fatal work |
| Device math functions (`tests/math_selftest`) | exp / log / tanh / sqrt / rsqrt / pow / identity: 0/1024 mismatches under the tool's combined tolerance (`abs <= 1e-6 + 1e-5*abs(ref)`); the largest absolute error is 4.883e-4 on `exp` at x=8.5, where the reference is 4914.77 (relative 1.19e-7) |
| Example executable | `gpt2 --help` loads and prints usage. **No model training step was run** |
| Absence of stale framework paths | the generated compile database contains no path to the previously used `5a7be78` source tree |

Note: Linear and Matmul currently execute through a **host fallback** - the operator tests above
prove that fallback path, not a device GEMM.

## Verified earlier (historical, pre-migration layout)

These were observed while the backend still lived under `mlu/` next to a separate framework tree.
They are recorded for traceability and were **not** re-executed for this layout change.

| Item | Historical evidence |
|---|---|
| Backend registration | the `--device mlu` validator accepted the backend and kernels dispatched on `kPrivateUse1` |
| Build dependency tracking | touching the launcher header rebuilt all `.mlu` objects; a following build did nothing |

## Not verified

| Item | Status |
|---|---|
| Standard recursive-submodule download | the validation build used an archive of the pinned framework tree plus shallow retrieval of the four dependencies at their pinned commits. File content matched the pinned revisions, but the plain `git submodule update --init --recursive` procedure was **not** exercised |
| Cross-provider regression | no MACA or CUDA build was run. The untouched official files are not evidence of cross-platform regression coverage |
| Device GEMM (CNNL) | not implemented; `src/kernels/linear.cc` does D2H -> host GEMM -> H2D and is marked `NOTE(MLU-FALLBACK)` |
| Full GPT-2 FP32 single-card 10 steps | **not completed**; earlier attempts hit a queue failure at large batch and a slow single-core backward |
| Backward coverage at model scale | partial; small-shape backward cases exist, full-model backward has not finished |
| BF16 | registered in dispatch only; `src/kernels/embedding.mlu` still rounds per token (marked `TODO(BF16-MUST-FIX)`), no BF16 accuracy run |
| Multi-card (CNCL) | not implemented (communication kernels absent) |
| Shared InfiniTrain test suites | not wired; `BUILD_TEST=ON` is rejected on purpose and the MLU regression tools are used instead |
| Performance | kernels run single task / serial (`cnrtFuncTypeBlock`); no multi-core or NRAM vectorisation yet |

## Known platform notes

- `__bang_atomic_add` triggered a queue error in our usage, so `EmbeddingBackward` accumulates
  with a row-partitioned scan instead (deterministic, no atomics).
- Device `powf` works once kernels run single-task/serial with `cnrtFuncTypeBlock`; with the
  earlier multi-task `cnrtFuncTypeUnion1` launch, loop writes to global memory did not persist.
- `MluGuardImpl::Memcpy` must copy through the current stream (`cnrtMemcpyAsync` + queue sync);
  plain `cnrtMemcpy` reads stale data while kernels are in flight.

## Next steps

1. device GEMM via CNNL (with the handle bound to the current stream) and re-run operator tests;
2. locate the large-shape queue failure, then run GPT-2 1 -> 2 -> 10 steps and compare loss
   step by step against a frozen A100 baseline;
3. BF16: FP32 accumulation in embedding, autocast/dtype promotion, softmax/LayerNorm/CE stability;
4. multi-card: CNCL registration and a 2-card DP milestone;
5. wire the shared InfiniTrain test suites and run a MACA/CUDA regression before claiming
   cross-provider compatibility.
