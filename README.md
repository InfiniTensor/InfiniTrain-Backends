# InfiniTrain Backends

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

Each provider is isolated under `backends/<provider>` and normally consumes a
pinned InfiniTrain submodule commit. One build tree selects one provider, and a
process may register at most one provider for `DeviceType::kPrivateUse1`.

## Supported Backends

| Backend | Runtime | Collectives | Model examples |
| ------- | ------- | ----------- | -------------- |
| MACA | MACA | MCCL (optional) | GPT-2, LLaMA 3, Mixtral |

The model sources normally come from the pinned InfiniTrain submodule. This
repository provides the provider-specific runtime, kernels, collective
implementation, and build integration needed to run them on MACA.

## Requirements

- Linux
- CMake 3.28 or newer
- Git with submodule support
- A compatible MACA SDK with a C++20-capable `mxgpu_llvm/bin/mxcc` compiler,
  the MACA runtime, MCDNN, and MCBLAS
- MCCL when distributed collectives are enabled
- `jq` when using the automated model test runner

## Quick Start

Initialize the pinned InfiniTrain submodule, select the MACA SDK, and build the
project:

```bash
git submodule update --init --recursive
export MACA_PATH=/opt/maca

mkdir build
cd build
cmake .. \
  -DINFINITRAIN_BACKEND=maca \
  -DINFINITRAIN_MACA_WITH_MCCL=ON \
  -DBUILD_TEST=ON
make -j
```

Top-level builds enable the backend examples by default. The example
executables are written to `build`:

```bash
./gpt2 --help
./llama3 --help
./mixtral --help
```

Run the registered MACA accelerator tests with CTest:

```bash
ctest -L maca --output-on-failure
```

## Training

As in InfiniTrain, each model example is an independent executable. Select the
MACA backend with `--device maca`. For example, a single-node LLaMA 3 training
run can be started from the build directory with:

```bash
./llama3 \
  --device maca \
  --input_bin [training_data_path] \
  --llmc_filepath [model_path] \
  --num_iteration 10
```

The GPT-2 and Mixtral executables follow the same command-line interface for
their corresponding model and dataset options. Run an executable with `--help`
to inspect all available options.

The model test matrix reuses InfiniTrain's test runner with MACA-specific
configuration. Update the dataset and checkpoint paths in
`backends/maca/scripts/test_config_maca.json` before running it from the
repository root in a separate shell:

```bash
backends/maca/scripts/run_models_and_profile.bash --only-run basic
```

## Build Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `INFINITRAIN_BACKEND` | Required | Provider selected from `backends/<provider>` |
| `INFINITRAIN_SOURCE_DIR` | `third_party/InfiniTrain` | InfiniTrain source tree, normally the pinned submodule |
| `INFINITRAIN_BACKENDS_BUILD_EXAMPLES` | `ON` for a top-level build | Build provider-enabled InfiniTrain examples |
| `BUILD_TEST` | `OFF` | Build InfiniTrain's full test set and MACA variants |
| `INFINITRAIN_MACA_WITH_MCCL` | `ON` | Enable MCCL distributed collectives |
| `MACA_PATH` | `$MACA_PATH` | MACA SDK root |

One build directory may contain only one provider. When working with multiple
providers, use a separate directory for each one so compiler and SDK cache
entries do not leak between them. Starting from the repository root:

```bash
mkdir build-maca
cd build-maca
cmake .. -DINFINITRAIN_BACKEND=maca
```

For development against another InfiniTrain checkout, override the pinned
submodule path explicitly:

```bash
mkdir build
cd build
cmake .. \
  -DINFINITRAIN_BACKEND=maca \
  -DINFINITRAIN_SOURCE_DIR=/path/to/InfiniTrain
```

The selected checkout must implement the PrivateUse1 extension API expected by
this backend.

To instantiate the shared accelerator tests, configure with `BUILD_TEST=ON`.
PrivateUse1 providers require `USE_CUDA=OFF`; configuration fails rather than
silently overriding an explicit `USE_CUDA=ON`. Their test identity is always
PrivateUse1, independent of the selected provider:

```bash
cmake .. -DBUILD_TEST=ON
cmake --build . --target test_tensor_maca test_autograd_maca
ctest -L maca --output-on-failure
```

The generated binaries are named `test_*_maca`, contain only
`PRIVATEUSE1/*` GTest instances, and carry the `maca`, `accelerator`, and
`hardware` CTest labels. The same build also contains InfiniTrain's CPU,
fake-provider, and CPU-only tests; CUDA remains disabled by the PrivateUse1
configuration contract. Use `ctest -L cpu` for the upstream CPU tests, or run
`ctest --output-on-failure` without `-L` to execute the complete registered set.

## Using the MACA Backend

Applications must register the provider before parsing or constructing a
`maca` device:

```cpp
#include "infini_train_maca/backend.h"

infini_train::maca::RegisterBackend();
auto type = infini_train::Device::ParseType("maca").value();
infini_train::Device device(type, 0);
```

`RegisterBackend()` installs the process-wide PrivateUse1 name, MACA kernels,
device guard, and, when enabled, the MCCL implementation. Registration itself
does not initialize the device runtime. MACA runtime initialization remains
lazy until InfiniTrain first requests the device guard.

Final executables should link `InfiniTrain::Backend::MACAExecutable` instead of
assembling the static archives themselves:

```cmake
add_executable(train main.cc)
target_link_libraries(train PRIVATE InfiniTrain::Backend::MACAExecutable)
```

This interface target retains InfiniTrain's static kernel-registration objects
and links the MACA provider in the required order. Library targets that do not
produce a final executable may link `InfiniTrain::Backend::MACA`.

## MACA Runtime Notes

- `MACA_LAUNCH_BLOCKING`: Unless already set by the user, the provider sets it
  to `1` immediately before lazy runtime initialization. Set the variable
  before the first device use to override it.
- `MCCL_P2P_DISABLE`: Immediately before runtime initialization, the provider
  reads `--tensor_parallel` from the process command line and sets this to `1`
  when the value is greater than one. An explicit environment value is
  preserved. Data-parallel jobs leave it unset to retain MCCL's P2P fast path.

## Architecture

```text
backends/<provider>/
  cmake/                 compiler and vendor SDK setup before project()
  include/               public provider API
  src/backend.cc         provider registration entry point
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

The MACA implementation uses InfiniTrain's existing registration mechanisms:
`REGISTER_KERNEL`, `INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL`, and
`INFINI_TRAIN_REGISTER_CCL_IMPL`. `RegisterBackend()` explicitly reaches the
runtime and kernel paths, plus the CCL path when MCCL is enabled. Static-library
object extraction is therefore driven by strong symbol references instead of
depending on global initialization order.

When this repository is embedded with `add_subdirectory()`, the parent project
must select `${MACA_PATH}/mxgpu_llvm/bin/mxcc` before its first `project()`
call. CMake cannot replace a compiler after a language has been enabled.

## License

InfiniTrain Backends is released under the [MIT License](LICENSE).
