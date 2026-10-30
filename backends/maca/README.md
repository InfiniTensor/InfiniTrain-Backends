# MACA Backend

**English** | [简体中文](README.zh-CN.md)

This backend provides MACA runtime support, kernels, and optional MCCL collectives
for InfiniTrain's GPT-2, LLaMA 3, and Mixtral examples.

See the [project README](../../README.md) for the project overview, shared build
options, and architecture. Unless otherwise noted, paths and commands below are
relative to the repository root.

## Requirements

In addition to the [common requirements](../../README.md#common-requirements):

- A compatible **MACA SDK** with a C++20-capable `mxgpu_llvm/bin/mxcc` compiler,
  the MACA runtime, MCDNN, MCBLAS, and the SDK's OpenMP runtime
- **MCCL** when distributed collectives are enabled (enabled by default)
- **GNU Make** for the quick-start build commands
- Python 3 when using the automated model test runner (standard library only)

## 🚀 Quick Start

Starting from the repository root, initialize the pinned InfiniTrain submodule,
select the MACA SDK, and build in a dedicated directory:

```bash
git submodule update --init --recursive
export MACA_PATH=/opt/maca

mkdir build
cd build
cmake .. -DINFINITRAIN_BACKEND=maca -DINFINITRAIN_MACA_WITH_MCCL=ON
make -j
```

The example executables are written to the current `build` directory. To view
their command-line options, run:

```bash
./gpt2 --help
./llama3 --help
./mixtral --help
```

For instructions on running specific test cases, see the
[InfiniTrain README](https://github.com/InfiniTensor/InfiniTrain#readme).

## MACA Build Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `INFINITRAIN_MACA_WITH_MCCL` | `ON` | Enable MCCL distributed collectives |
| `MACA_PATH` | `$MACA_PATH` | MACA SDK root |

`MACA_PATH` can be set through the environment or with `-DMACA_PATH=/path/to/maca`.
To build without MCCL, pass `-DINFINITRAIN_MACA_WITH_MCCL=OFF`.

## 🏋️ Training

Each model example is an independent executable. Select the
MACA backend with `--device maca`. For example, a single-node LLaMA 3 training
run can be started from the build directory with:

```bash
./llama3 \
  --device maca \
  --input_bin /path/to/training_data.bin \
  --llmc_filepath /path/to/model.bin \
  --num_iteration 10
```

GPT-2 and LLaMA 3 use `--batch_size` and `--total_batch_size`; Mixtral
uses `--micro_batch_size` and `--global_batch_size`. Run an executable with
`--help` to inspect its model and dataset options.

Model tests cover combinations of models, training precision, and parallel
configurations. They reuse InfiniTrain's test runner and
`third_party/InfiniTrain/scripts/test_config.json`. The MACA entry point merges
that configuration with the overrides in
`backends/maca/scripts/test_config_maca.json`. All upstream groups and tests are
inherited in their original order. Entries in the overrides' `test_groups` only
customize settings; groups and tests omitted there are inherited as well.
Before running it, update the dataset and checkpoint paths in the overrides.
The wrapper uses Python's standard library for JSON handling, including the
upstream runner's configuration queries, so `jq` is not required.
The runner uses `build/maca` by default and clears that directory before each
rebuild and after the run.

Run all configured model tests from the repository root in a separate shell
with the MACA SDK environment configured:

```bash
backends/maca/scripts/run_models_and_profile.bash
```

Without `--only-run`, the runner executes all enabled test groups in the
configuration. To run just the basic group, add `--only-run basic`.

## Unit Tests

Unit tests are disabled by default. From the `build` directory created in the
quick start, enable and build them with:

```bash
cmake .. -DBUILD_TEST=ON
make -j
ctest -L maca --output-on-failure
```

The generated MACA unit test binaries are named `test_*_maca`, contain only
`PRIVATEUSE1/*` GTest instances, and carry the `maca` CTest label. This label
identifies the backend and can be selected with CTest's `-L` option.

The same build also contains InfiniTrain's CPU, fake-provider, and CPU-only
tests. Use `ctest -L cpu` for the upstream CPU tests, or run
`ctest --output-on-failure` without `-L` to execute the complete registered set.

## Application Integration

Applications must register the provider before parsing or constructing a
`maca` device:

```cpp
#include "infini_train/include/device.h"
#include "infini_train_maca/backend.h"

infini_train::maca::RegisterBackend();
auto type = infini_train::Device::ParseType("maca").value();
infini_train::Device device(type, 0);
```

`RegisterBackend()` registers the process-wide PrivateUse1 name `maca` and
sets its default autocast dtype to BF16. Kernels, the device guard, and optional
MCCL support register through static initialization; the device runtime remains
lazy until InfiniTrain first requests the device guard.

Final executables should link `InfiniTrain::Backend::MACAExecutable` instead of
assembling the static archives themselves:

```cmake
add_executable(train main.cc)
target_link_libraries(train PRIVATE InfiniTrain::Backend::MACAExecutable)
```

This interface target uses `--whole-archive` to retain static registration
objects from InfiniTrain's core, CPU kernels, and the MACA provider. Library
targets that do not produce a final executable may link
`InfiniTrain::Backend::MACA`.

The MACA implementation uses InfiniTrain's existing registration mechanisms:
`REGISTER_KERNEL`, `INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL`, and
`INFINI_TRAIN_REGISTER_CCL_IMPL`. These registrations run during static
initialization; `RegisterBackend()` registers the provider metadata before
`maca` device parsing.

When adding the MACA backend as a third-party library with `add_subdirectory()`,
select `${MACA_PATH}/mxgpu_llvm/bin/mxcc` in the parent project before enabling
C++.

## Runtime Notes

Configure these independent boolean flags in each model test's `args` in
`backends/maca/scripts/test_config_maca.json`, or pass them directly to the
example executable. The multithread workaround defaults to `true`, pool
retention defaults to `false`, and the runtime does not infer a mode from the
thread or process count.

| Flag | Default | Behavior |
| ---- | ------- | -------- |
| `--maca_multithread_workarounds=true\|false` | `true` | Enables synchronous allocation/free, the copy mutex, default launch blocking, and the existing example synchronization/exit workarounds. |
| `--maca_retain_async_pool=true\|false` | `false` | Retains pages in the SDK default async pool for reuse by setting its release threshold to `UINT64_MAX`. |

The JSON omits redundant `maca_multithread_workarounds=true` entries and
explicitly disables that flag for cases using the native async path: basic
cases `1`, `1_bfloat16`, `2`, `2_bfloat16`, and every `8_proc` case. It enables
pool retention for those same cases. Pool retention avoids the reproduced
MACA 3.5.3.18 ATU fault in mixed parallel training by keeping idle pages
available for reuse; explicit trimming can reintroduce the fault.

Parse flags before first device use. The runtime captures their values once.
Explicit `MACA_LAUNCH_BLOCKING` and `MCCL_P2P_DISABLE` environment values retain
precedence. When workarounds are enabled, the provider defaults
`MACA_LAUNCH_BLOCKING` to `1` and also defaults `MCCL_P2P_DISABLE` to `1` when
the command line specifies `--tensor_parallel` greater than `1`. With
workarounds disabled it sets neither
variable; multiprocess TP therefore needs separate validation with the SDK's
native P2P behavior. Example workarounds stay in InfiniTrain and are selected
by one optional build-time flag binding. Large MCCL all-reduce messages are
segmented at 160 MiB by default; set `INFINI_MCCL_ALLREDUCE_SEGMENT_MB=0` to
disable segmentation.
