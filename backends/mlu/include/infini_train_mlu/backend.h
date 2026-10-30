#pragma once

// MLU（寒武纪）后端的注册入口。上游示例程序通过编译定义
// INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER / _REGISTRAR 在解析命令行
// 之前调用 RegisterBackend()（必须在 gflags 校验 --device 之前完成注册，
// 因为 Device::ParseType("mlu") 依赖注册后的 backend 名字）。

namespace infini_train::backends::mlu {

void RegisterBackend();

} // namespace infini_train::backends::mlu
