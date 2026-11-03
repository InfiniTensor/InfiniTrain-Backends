#!/usr/bin/env bash
# 设备数学函数单测的编译 + 运行脚本（在 MLU 服务器/容器的本目录下执行）
#
# 用法：bash build_and_run.sh
# 依赖：cncc、cnrt（NEUWARE_HOME，默认 /usr/local/neuware）
# 说明：默认不带 --bang-mlu-arch（与框架 CMake 的做法一致）；需要指定时
#       MLU_ARCH=mtp_613 bash build_and_run.sh
set -euo pipefail

NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
ARCH="${MLU_ARCH:-}"

echo "[1/3] cncc 编译 .mlu"
NEUWARE_HOME="$NEUWARE_HOME" "$NEUWARE_HOME/bin/cncc" -std=c++17 -fPIC -c math_selftest.mlu \
  -o math_selftest.o -I. -I"$NEUWARE_HOME/include" ${ARCH:+--bang-mlu-arch=$ARCH}

echo "[2/3] g++ 编译 host"
g++ -std=c++17 -O2 main.cc math_selftest.o \
  -I"$NEUWARE_HOME/include" -L"$NEUWARE_HOME/lib64" -lcnrt -o math_selftest

if [[ "${1:-}" == "--build-only" ]]; then echo "[3/3] 跳过运行（--build-only）"; exit 0; fi

echo "[3/3] 运行"
LD_LIBRARY_PATH="$NEUWARE_HOME/lib64:${LD_LIBRARY_PATH:-}" ./math_selftest
