# bang_host 头文件（vendor 拷贝）

`bang_fp16.h` / `bang_bf16.h` 拷自 Neuware 安装：

    /usr/local/neuware/lib/clang/11.1.0/include/   (Neuware CNRT 7.7.3 / cncc 5.7.6)

## 为什么拷贝

host 侧（g++）编译单元需要 `__half` / `bfloat16_t` 类型定义
（`mlu/runtime/mlu_dispatch.h`），但这两个头只在 cncc 的 clang 资源目录里，
该目录同时含有 clang 专用的 `stddef.h` 等内建头。若把整个目录加进 g++ 的
include 路径，clang 的 `stddef.h` 会遮蔽 GCC 自带版本（`__has_feature` 是
clang 专有扩展，GCC 预处理器直接报错），导致所有 host 编译单元失败。

这两个头本身是自包含的（`bang_bf16.h` 仅 include `bang_fp16.h`，后者无
include），host 路径（`__GNUC__` 分支）不依赖任何 clang 内建头，因此拷贝
出来单独使用是安全的。

`.mlu` 设备代码不受影响：cncc 自动从自己的资源目录找到 `<bang.h>` 等头，
无需 `-I`。

## 许可证

两头部为 Cambricon 的 permissive license（见文件头注释，允许拷贝/修改/
再分发，需保留版权声明）。

## 本地修改（与原版的唯一差异）

host 分支（非 `__BANGCC__`）的 `__BANG_HOSTDEVICE__` /
`__BANG_HOSTDEVICE_BF16__` 宏从空改为 `inline`：原版假设 host 侧只有
一个编译单元包含这些头，多 TU 包含时全部 free operator / 成员函数定义
会 multiple definition 链接错误；设备（cncc）分支本就带 `inline`，此改动
只是让 host 分支与设备分支一致。

## 更新时机

Neuware 大版本升级后，对比原目录中的同名文件，有变化则同步更新。
host 侧只用到位布局稳定的 16 位浮点类型定义，版本漂移风险很低。
