# InfiniTrain Backends

**English** | [简体中文](README.zh-CN.md)

[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/issues
)
[![PR](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/pulls
)
[![License](https://img.shields.io/github/license/InfiniTensor/InfiniTrain-Backends)](
https://github.com/InfiniTensor/InfiniTrain-Backends/blob/master/LICENSE
)

InfiniTrain Backends provides out-of-tree accelerator backends for
[InfiniTrain](https://github.com/InfiniTensor/InfiniTrain). It keeps
vendor-specific SDK integration, runtime support, collective communication,
and kernels outside the framework core while implementing InfiniTrain's
`PrivateUse1` backend interfaces.

Each provider is isolated under `backends/<provider>` and normally consumes the
InfiniTrain submodule pinned to `v0.7.0`. One build tree selects one provider, and a
process may register at most one provider for `DeviceType::kPrivateUse1`.

## Supported Backends

| Backend | Runtime | Collectives | Model examples |
| ------- | ------- | ----------- | -------------- |
| [MACA](#maca) | MACA | MCCL (optional) | GPT-2, LLaMA 3, Mixtral |

The model sources normally come from the pinned InfiniTrain submodule. Each
backend supplies the runtime, kernels, collective implementation, and build
integration needed to run its supported examples on the corresponding
accelerator. See the backend guides below for setup and usage.

## Common Requirements

- **Linux**
- **CMake**: 3.28 or newer
- **Git** with submodule support
- A **C++20** compiler and vendor SDK compatible with the selected backend

Additional dependencies and SDK setup are documented in each backend's guide.

## Build Options

These options apply across backends. Provider-specific options are documented
in the corresponding backend guide.

| Option | Default | Description |
| ------ | ------- | ----------- |
| `INFINITRAIN_BACKEND` | Required | Provider selected from `backends/<provider>` |
| `INFINITRAIN_SOURCE_DIR` | `third_party/InfiniTrain` | InfiniTrain source tree, normally the pinned submodule |
| `INFINITRAIN_BACKENDS_BUILD_EXAMPLES` | `ON` for a top-level build | Build provider-enabled InfiniTrain examples |
| `BUILD_TEST` | `OFF` | Build InfiniTrain's unit tests and variants for the selected provider |

Use a separate build directory for each provider so compiler and SDK cache
entries do not leak between them. Top-level builds enable backend examples by
default.

For development against another InfiniTrain checkout, add
`-DINFINITRAIN_SOURCE_DIR=/path/to/InfiniTrain` to the configure command. The
selected checkout must implement the PrivateUse1 extension API expected by
the backend.

PrivateUse1 providers require `USE_CUDA=OFF`; configuration fails rather than
silently overriding an explicit `USE_CUDA=ON`. Unit tests for the selected
provider use the PrivateUse1 device type and `PRIVATEUSE1/*` GTest prefix,
independent of the provider's name.

## Architecture

```text
backends/<provider>/
  cmake/                 compiler and vendor SDK setup before project()
  include/               public provider API
  src/backend.cc         provider metadata registration entry point
  src/common/            provider-internal shared helpers
  src/runtime/           device, stream, event, and allocator integration
  src/kernels/           provider kernel implementations and registration
  src/ccl/               optional collective communication integration
  examples/              CMake adapters for upstream InfiniTrain examples
  scripts/               provider model-run configuration and wrappers

third_party/InfiniTrain/ framework source, normally a pinned git submodule
```

The root build selects and includes
`backends/<provider>/cmake/pre_project.cmake` before its first `project()` call.
This lets each provider select its compiler and prepare SDK-specific dependency
probes without adding vendor branches to the root `CMakeLists.txt`.

When adding this repository as a third-party library with `add_subdirectory()`,
select the compiler required by the provider in the parent project before
enabling the C++ language. After setting up the vendor SDK environment, use
the following order in the parent's `CMakeLists.txt`, replacing the compiler
path and provider name:

```cmake
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_COMPILER "/path/to/vendor/compiler" CACHE FILEPATH "")
set(INFINITRAIN_BACKEND "<provider>" CACHE STRING "")

project(MyTrainingApp LANGUAGES CXX)
add_subdirectory(third_party/InfiniTrain-Backends)
```

CMake cannot replace a compiler after a language has been enabled.

## Backend Guides

Each backend has its own section for dependencies, build commands, examples,
and runtime behavior.

### MACA

#### Requirements

In addition to the common requirements:

- A compatible **MACA SDK** with a C++20-capable `mxgpu_llvm/bin/mxcc` compiler,
  the MACA runtime, MCDNN, MCBLAS, and the SDK's OpenMP runtime
- **MCCL** when distributed collectives are enabled (enabled by default)
- Python 3 when using the automated model test runner (standard library only)

#### 🚀 Quick Start

Starting from the repository root, initialize the pinned InfiniTrain submodule,
select the MACA SDK, and build in a dedicated directory:

```bash
git submodule update --init --recursive
export MACA_PATH=/opt/maca

cmake -S . -B build-maca \
  -DINFINITRAIN_BACKEND=maca \
  -DINFINITRAIN_MACA_WITH_MCCL=ON \
  -DBUILD_TEST=ON
cmake --build build-maca -j
```

The example executables are written to `build-maca`. From that directory:

```bash
cd build-maca
./gpt2 --help
./llama3 --help
./mixtral --help
```

#### MACA Build Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `INFINITRAIN_MACA_WITH_MCCL` | `ON` | Enable MCCL distributed collectives |
| `MACA_PATH` | `$MACA_PATH` | MACA SDK root |

`MACA_PATH` can be set through the environment or with `-DMACA_PATH=/path/to/maca`.
To build without MCCL, pass `-DINFINITRAIN_MACA_WITH_MCCL=OFF`.

#### 🏋️ Training

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
`backends/maca/scripts/test_config_maca.json`. Before running it, update the
dataset and checkpoint paths in the overrides.
The wrapper uses Python's standard library for JSON handling, including the
upstream runner's configuration queries, so `jq` is not required.

Run all configured model tests from the repository root in a separate shell
with the MACA SDK environment configured:

```bash
backends/maca/scripts/run_models_and_profile.bash
```

Without `--only-run`, the runner executes all enabled test groups in the
configuration. To run just the basic group, add `--only-run basic`.

#### Unit Tests

The quick start builds the unit tests with `BUILD_TEST=ON`. To enable and build
them in an existing MACA build directory, run:

```bash
cmake .. -DBUILD_TEST=ON
cmake --build . -j
ctest -L maca --output-on-failure
```

The generated MACA unit test binaries are named `test_*_maca`, contain only
`PRIVATEUSE1/*` GTest instances, and carry the `maca` CTest label. This label
identifies the backend and can be selected with CTest's `-L` option.

The same build also contains InfiniTrain's CPU, fake-provider, and CPU-only
tests. Use `ctest -L cpu` for the upstream CPU tests, or run
`ctest --output-on-failure` without `-L` to execute the complete registered set.

#### Application Integration

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

#### Runtime Notes

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

## License

InfiniTrain Backends is released under the [MIT License](LICENSE).
