# 代码格式和静态检查

这些脚本及 `tools/clang_tidy` 插件迁移自 `third_party/InfiniTrain`，
以下命令供后续手动使用。Python 要求 3.10 或更新版本。
相对的源码、构建目录、插件和报告路径均以本仓库根目录为基准。

## 迁移适配

| 项目 | Backends 中的行为 |
| --- | --- |
| 源码范围 | clang-tidy/IWYU 只选择 `backends/`、`tools/` 中属于本仓库的宿主编译单元；排除 `third_party`、构建目录和预期编译失败的 `*_compile_fail.cc` |
| 设备源码 | 格式化支持 `.cu`、`.maca`、`.mlu`、`.cl`；clang-tidy/IWYU 排除这些编译单元，并输出排除数量 |
| 头文件变更 | 检查编译数据库中全部宿主编译单元，覆盖间接引用；不再只选一个直接引用者 |
| 构建、依赖或检查规则变更 | 检查全部宿主编译单元；CMake 变更后需先重新生成编译数据库 |
| 编译数据库 | 派生过滤后的数据库，保留原编译参数；没有本仓库宿主条目或变更的宿主源文件缺少条目时均报错，需启用相关目标并重新生成数据库 |
| 格式化 | 沿用根目录 `.clang-format` 的 4 空格、120 列；修正失败返回码及 Black 成功输出被误判的问题，去除 `colorama` 依赖 |
| 自定义 tidy 规则 | 保留 InfiniTrain 的规则名和插件名，插件独立于后端构建，要求 LLVM/Clang 18 |

## clang-format 与 Black

`clang_format_diff.py` 要求 **clang-format 16.0.6**，默认查找
`clang-format-16`，也可通过 `--clang-format /path/to/clang-format` 指定。
默认只检查或格式化暂存区变更行，`--ref` 使用指定引用与 HEAD 的 merge-base
到 HEAD 的差异，`--all` 选择本仓库全部已跟踪的 C/C++ 和设备源码。
依赖、构建目录及删除的文件均会跳过。

```bash
# 检查暂存区变更行；省略 --check 会写回格式化结果
python3 scripts/clang_format_diff.py --check
python3 scripts/clang_format_diff.py --check --ref HEAD~1
python3 scripts/clang_format_diff.py --check --all

# 原有入口：检查整个文件，支持 C/C++、设备源码与 Python
python3 scripts/format.py --path backends tools scripts --check
```

`format.py` 保留 `--ref`、`--path`、`--check`、`--c`、`--py` 参数。
不传 `--ref` 和 `--path` 时选择暂存区中的文件；`--ref` 与上述脚本一致；
`--path` 单独使用时递归选择文件，也包含未跟踪文件。
`--c` 默认 `clang-format-16`，`--py` 默认 `black`，均支持二进制绝对路径。
检查使用返回码判断结果，`--check` 不会写回文件。

暂存区模式与 `--ref` 模式都基于差异选择范围，工具读取工作区文件。
使用时应让所选文件的工作区内容与被比较的版本一致，避免部分暂存导致行号偏移。

## 后端编译数据库

clang-tidy 和 IWYU 需要仓库根目录生成的 `compile_commands.json`。
仅在 InfiniTrain 子目录开启导出时，数据库可能只有子模块条目，无法检查后端。
MACA 示例配置如下，其他构建依赖按 [MACA 指南](../backends/maca/README.zh-CN.md) 准备：

```bash
cmake -S . -B build/lint -G Ninja \
  -DINFINITRAIN_BACKEND=maca \
  -DMACA_PATH=/opt/maca \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DINFINITRAIN_BACKENDS_BUILD_EXAMPLES=OFF \
  -DBUILD_TEST=OFF
```

这里使用后端的真实编译器及 SDK。不要沿用上游 CPU CI 的纯 clang 配置来
代替后端配置。保留独立的 `build/lint` 目录，避免与模型运行脚本清理的构建目录混用。
多个后端需分别配置和检查；`--all` 只覆盖当前数据库中存在的宿主编译单元。

运行检查脚本时，会在 `BUILD_DIR/clang-tidy-host` 或 `BUILD_DIR/iwyu-host`
写入过滤后的数据库，原数据库保持完整。编译参数和包含路径不会被删改。

## clang-tidy

`.clang-tidy` 启用两个规则：非模板、非 constexpr 成员函数应移出头文件
（空函数体等情况豁免），以及枚举常量使用 `kFooBar` 命名。
迁移保留了这些上游规则；首次检查可能发现已有代码违反规则。

自定义规则需要 `InfiniTrainTidy.so`，必须与 clang-tidy 来自同一套 LLVM/Clang 18
开发环境。不要用 MACA 的 mxcc 构建此插件。典型的独立构建方式如下，
LLVM/Clang 的 CMake 路径需按实际安装位置调整：

```bash
cmake -S tools/clang_tidy -B build/clang-tidy-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build/clang-tidy-plugin

python3 scripts/clang_tidy.py --all \
  --build-dir build/lint \
  --clang-tidy clang-tidy-18 \
  --run-clang-tidy run-clang-tidy-18 \
  --load-plugin build/clang-tidy-plugin/InfiniTrainTidy.so
```

不传 `--all` 时选择暂存区变更，或使用 `--ref HEAD~1`。
仅宿主源文件变化时，直接对选中的编译单元运行 clang-tidy 并限制诊断到变更行；
纯删除没有新增行范围时检查该文件全部内容。头文件、CMake、子模块、规则或插件变化
会改为全部宿主检查，因此也需要 `run-clang-tidy-18`。
迁移后的增量模式不再依赖 `clang-tidy-diff`，避免将没有编译命令的头文件交给它。

`--clang-tidy` 指定的二进制也会传给全量 runner。
`--jobs` 控制并发；全量扫描在 `BUILD_DIR/clang-tidy-report` 输出原始和去重诊断，
可通过 `--report-dir` 改位置。脚本不会应用 tidy 自动修复。

## IWYU

需要支持 C++20 的 `include-what-you-use`、`iwyu_tool.py`，修复模式另需
`fix_includes.py`；这些工具应来自同一套安装。上游使用的 LLVM 17 环境不能直接
证明与当前 MACA SDK 兼容。`scripts/iwyu.imp` 保留标准库与 Eigen 的映射，
没有添加未经验证的 MACA SDK 映射。

```bash
python3 scripts/iwyu.py --all --build-dir build/lint
python3 scripts/iwyu.py --ref HEAD~1 --build-dir build/lint
# 只预览 include 修复
python3 scripts/iwyu.py --all --build-dir build/lint --fix-dry-run
```

默认选择暂存区变化的宿主源文件；头文件、CMake、子模块或 IWYU 规则变化时选择
全部宿主编译单元。默认只检查，发现 include 问题返回非零。
`--fix` 才会改文件，修复范围限制在本仓库的 `backends`、`tools` 下，
默认保留头文件 include；`--fix-header-removals` 才允许删除它们。
可用 `--iwyu-tool` 和 `--fix-tool` 指定工具路径。

两个静态检查入口都支持 `--has-relevant-changes [--ref REF]`：
有相关变更返回 0，没有返回 1，读取失败返回 2；此模式不运行分析器。

## 尚需实际验证的适配

MACA 将 `.maca` 作为 CXX 并传入 `-x maca`，其中包含设备关键字、内核启动语法和
厂商内建函数。普通 clang-tidy/IWYU 无法按此语言解析，因此脚本不将其强制当普通
C++ 检查。设备文件及仅由它们引用的头文件仍需 SDK 支持的分析方式。

即便是宿主 `.cc`，MACA SDK 头文件、编译器隐式包含路径及厂商参数与 LLVM 18/IWYU
前端的兼容性也需要实际检查。若报 unknown argument、缺少 SDK 头文件或内建类型，
应先核对 SDK 和分析前端，而不是忽略错误或删除语言参数来获得通过结果。
编译数据库过滤不代表这些兼容性问题已解决。

未迁移上游的 clang-tidy/IWYU 自动 CI：它们只配置 CPU 构建，不满足此仓库的
后端及 SDK 要求。现有格式化 CI 仍只检查 `backends`，显式安装 clang-format 16.0.6
和 Black；新增分析器由开发者手动运行。
