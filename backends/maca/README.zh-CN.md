# MACA 后端

[English](README.md) | **简体中文**

本后端为 InfiniTrain 的 GPT-2、LLaMA 3 和 Mixtral 示例提供 MACA 运行时支持、
算子和可选的 MCCL 集合通信。

项目介绍、通用构建选项和架构见[项目 README](../../README.zh-CN.md)。
除非另有说明，以下路径和命令均以仓库根目录为基准。

## 依赖

除[通用依赖](../../README.zh-CN.md#通用依赖)外，还需要：

- 兼容的 **MACA SDK**，包含支持 C++20 的 `mxgpu_llvm/bin/mxcc` 编译器、MACA
  运行时、MCDNN、MCBLAS 及 SDK 自带的 OpenMP 运行时
- 启用分布式集合通信时需要 **MCCL**（默认启用）
- 快速开始中的构建命令需要 **GNU Make**
- 使用自动化模型测试脚本时需要 Python 3（仅使用标准库）

## 🚀 快速开始

从仓库根目录执行以下命令，初始化锁定版本的 InfiniTrain 子模块、选择 MACA SDK，
并在独立目录中构建：

```bash
git submodule update --init --recursive
export MACA_PATH=/opt/maca

mkdir build
cd build
cmake .. -DINFINITRAIN_BACKEND=maca -DINFINITRAIN_MACA_WITH_MCCL=ON
make -j
```

示例可执行文件输出到当前的 `build` 目录。可直接查看各示例的命令行选项：

```bash
./gpt2 --help
./llama3 --help
./mixtral --help
```

具体测例的运行方式请参考 [InfiniTrain README](https://github.com/InfiniTensor/InfiniTrain#readme)。

## MACA 构建选项

| 选项 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `INFINITRAIN_MACA_WITH_MCCL` | `ON` | 启用 MCCL 分布式集合通信 |
| `MACA_PATH` | `$MACA_PATH` | MACA SDK 根目录 |

`MACA_PATH` 可通过环境变量或 `-DMACA_PATH=/path/to/maca` 设置。
如需在不使用 MCCL 的情况下构建，可传入 `-DINFINITRAIN_MACA_WITH_MCCL=OFF`。

## 🏋️ 训练

各模型示例都是独立的可执行文件。通过 `--device maca`
选择 MACA 后端。例如，在构建目录中执行以下命令可启动单节点 LLaMA 3 训练：

```bash
./llama3 \
  --device maca \
  --input_bin /path/to/training_data.bin \
  --llmc_filepath /path/to/model.bin \
  --num_iteration 10
```

GPT-2 和 LLaMA 3 使用 `--batch_size` 与 `--total_batch_size`；Mixtral 使用
`--micro_batch_size` 与 `--global_batch_size`。可通过各可执行文件的 `--help`
查看模型及数据集相关选项。

模型测试覆盖不同模型、训练精度和并行配置的组合，复用 InfiniTrain 的测试脚本和
`third_party/InfiniTrain/scripts/test_config.json`。MACA 入口脚本将该配置与
`backends/maca/scripts/test_config_maca.json` 中的覆盖配置合并，继承上游全部测试组
及组内测例，并保留上游顺序。覆盖配置中的 `test_groups` 仅用于修改参数；未列出的组
和测例也会继承。运行前，请修改
覆盖配置中的数据集和 checkpoint 路径。
入口脚本使用 Python 标准库处理 JSON，包括上游脚本的配置查询，无需安装 `jq`。
自动测试默认在 `build/maca` 中构建，脚本会在每次重新构建前和运行结束后清理该目录。

请在已配置 MACA SDK 环境的另一个 shell 中，从仓库根目录运行全部已配置的模型
测试：

```bash
backends/maca/scripts/run_models_and_profile.bash
```

不传 `--only-run` 时，脚本会运行配置中启用的全部测试组。如只需运行 basic 组，
可添加 `--only-run basic`。

## 单元测试

单元测试默认关闭。在快速开始创建的 `build` 目录中，执行以下命令启用并构建
单元测试：

```bash
cmake .. -DBUILD_TEST=ON
make -j
ctest -L maca --output-on-failure
```

生成的 MACA 单元测试程序命名为 `test_*_maca`，仅包含 `PRIVATEUSE1/*` GTest
实例，并带有 `maca` CTest 标签。该标签标识后端，可通过 CTest 的 `-L` 选项筛选。

同一构建中还包含 InfiniTrain 的 CPU 测试、模拟后端测试和仅适用于 CPU 的测试。
使用 `ctest -L cpu` 可运行上游 CPU 测试；使用不带 `-L` 的
`ctest --output-on-failure` 可运行全部已注册测试。

## 应用集成

应用应在解析或构造 `maca` 设备前注册后端：

```cpp
#include "infini_train/include/device.h"
#include "infini_train_maca/backend.h"

infini_train::maca::RegisterBackend();
auto type = infini_train::Device::ParseType("maca").value();
infini_train::Device device(type, 0);
```

`RegisterBackend()` 注册进程级 PrivateUse1 名称 `maca`，并将其默认 autocast
数据类型设为 BF16。算子、设备 guard 和可选的 MCCL 支持通过静态初始化注册；
设备运行时延迟到 InfiniTrain 首次请求设备 guard 时才初始化。

最终可执行文件应链接 `InfiniTrain::Backend::MACAExecutable`，由该目标处理静态库
的链接：

```cmake
add_executable(train main.cc)
target_link_libraries(train PRIVATE InfiniTrain::Backend::MACAExecutable)
```

该接口目标使用 `--whole-archive` 保留 InfiniTrain 核心、CPU 算子和 MACA 后端的
静态注册对象。不生成最终可执行文件的库目标可链接 `InfiniTrain::Backend::MACA`。

MACA 实现使用 InfiniTrain 现有的注册机制：`REGISTER_KERNEL`、
`INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL` 和 `INFINI_TRAIN_REGISTER_CCL_IMPL`。
这些注册在静态初始化阶段执行；`RegisterBackend()` 在解析 `maca` 设备前注册
后端元数据。

通过 `add_subdirectory()` 将 MACA 后端作为第三方库添加到父项目时，应在启用 C++
前选择 `${MACA_PATH}/mxgpu_llvm/bin/mxcc` 编译器。

## 运行时说明

以下两个布尔参数相互独立，可在 `backends/maca/scripts/test_config_maca.json`
中各模型测试的 `args` 中配置，也可直接传给示例可执行文件。多线程兼容措施默认为
`true`，异步内存池页面保留默认为 `false`，运行时不会根据线程数或进程数推断模式。

| 参数 | 默认值 | 行为 |
| ---- | ------ | ---- |
| `--maca_multithread_workarounds=true\|false` | `true` | 启用同步分配与释放、内存复制互斥锁、默认的阻塞式 launch，以及现有示例中的同步和退出兼容措施。 |
| `--maca_retain_async_pool=true\|false` | `false` | 将 SDK 默认异步内存池的释放阈值设为 `UINT64_MAX`，保留页面以便复用。 |

JSON 配置省略了冗余的 `maca_multithread_workarounds=true`，并对使用原生异步
路径的用例显式关闭该参数：包括 basic 组中的 `1`、`1_bfloat16`、`2`、
`2_bfloat16`，以及所有 `8_proc` 用例。这些用例同时启用了内存池页面保留。
在混合并行训练中，保留空闲页面供后续复用，可避开已复现的 MACA 3.5.3.18 ATU
故障；显式裁剪内存池可能再次触发该故障。

请在首次使用设备前解析参数，运行时只捕获一次参数值。显式设置的
`MACA_LAUNCH_BLOCKING` 和 `MCCL_P2P_DISABLE` 环境变量优先。在启用兼容措施时，
后端默认将 `MACA_LAUNCH_BLOCKING` 设为 `1`；如果命令行指定的
`--tensor_parallel` 大于 `1`，还默认将 `MCCL_P2P_DISABLE` 设为 `1`。关闭兼容
措施时，后端不会设置这两个变量，因此多进程 TP 仍需单独验证 SDK 的原生 P2P
行为。示例中的兼容措施保留在 InfiniTrain 中，通过一个可选的构建期参数绑定来
控制。较大的 MCCL all-reduce 消息默认按 160 MiB 分段；设置
`INFINI_MCCL_ALLREDUCE_SEGMENT_MB=0` 可禁用分段。
