// 逐算子对拍 + 比较器自检 + 探针用例（001-P2 v2 R01/R02/R05/R06）。
//
// 判据（R01 修正后）：
//   1) 先按显式期望分类：ref 为 NaN/±Inf 时要求 got 同类（NaN↔NaN、同号 Inf），
//      且只有声明"允许特殊参考"的域用例才接受 ref 非有限；
//   2) ref 有限时 got 必须有限，再用 |got-ref| <= atol + rtol*|ref|；
//   3) 普通用例出现非预期 NaN/Inf（哪怕两侧相同）一律失败。
//
// 模式（解析在 backend 注册与设备初始化之前完成）：
//   （默认）            设备对拍套件；有失败 → 退出 1
//   --comparator-selftest  纯 host，逐个断言比较器判断，全部符合才退出 0
//   --inject-finite     设备对拍并注入有限误差，预期整轮失败（退出非 0，由外层脚本检查）
//   --inject-nan        设备对拍并注入非预期 NaN，预期整轮失败
//   --probe-case NAME   设备上的探针用例（只分配+拷贝，不发 kernel），配合 MLU_DEBUG_CHECK=1 由脚本断言日志
//   --device N          选择设备

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train_mlu/backend.h"
#include "runtime/mlu_debug.h"

namespace {

using infini_train::DataType;
using infini_train::Device;
using infini_train::Tensor;

constexpr cnrtRet_t CNRT_RET_SUCCESS_LOCAL = cnrtSuccess;
constexpr auto kCPU = Device::DeviceType::kCPU;
constexpr auto kMLU = Device::DeviceType::kPrivateUse1;

bool g_failed = false;
bool g_inject_finite = false;
bool g_inject_nan = false;
int g_device = 0;

Device Cpu() { return Device(kCPU, 0); }
Device Mlu() { return Device(kMLU, g_device); }

std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &host, Device dev) {
    auto t = std::make_shared<Tensor>(dims, DataType::kFLOAT32, dev);
    if (dev.type() == kCPU) {
        std::memcpy(t->DataPtr(), host.data(), host.size() * sizeof(float));
        return t;
    }
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), host.data(), host.size() * sizeof(float), infini_train::core::MemcpyKind::kH2D);
    return t;
}

std::shared_ptr<Tensor> MakeTensorI64(const std::vector<int64_t> &dims, const std::vector<int64_t> &host, Device dev) {
    auto t = std::make_shared<Tensor>(dims, DataType::kINT64, dev);
    if (dev.type() == kCPU) {
        std::memcpy(t->DataPtr(), host.data(), host.size() * sizeof(int64_t));
        return t;
    }
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), host.data(), host.size() * sizeof(int64_t), infini_train::core::MemcpyKind::kH2D);
    return t;
}

std::vector<float> ToHost(const std::shared_ptr<Tensor> &t) {
    std::vector<float> host(t->NumElements());
    if (t->GetDevice().type() == kCPU) {
        std::memcpy(host.data(), t->DataPtr(), host.size() * sizeof(float));
        return host;
    }
    infini_train::core::GetDeviceGuardImpl(t->GetDevice().type())
        ->Memcpy(host.data(), t->DataPtr(), host.size() * sizeof(float), infini_train::core::MemcpyKind::kD2H);
    return host;
}

std::vector<float> RandVec(size_t n, float scale, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, scale);
    std::vector<float> v(n);
    for (auto &x : v) { x = dis(gen); }
    return v;
}

// ---------------------------------------------------------------------------
// 比较器（R01）：非有限优先分类，只有两侧都有限时才用 atol+rtol
// ---------------------------------------------------------------------------
struct Verdict {
    size_t bad = 0;
    size_t worst = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool empty() const { return bad == 0; }
};

Verdict Evaluate(const std::vector<float> &ref, const std::vector<float> &got, double atol, double rtol,
                 bool allow_special_ref) {
    Verdict v;
    if (ref.size() != got.size()) {
        v.bad = 1;
        return v;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = got[i];
        const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
        const bool a_inf = std::isinf(a), b_inf = std::isinf(b);

        if (a_nan || a_inf) {
            // 参考本身非有限：只有声明允许时才可接受，且必须同类同号
            const bool same = (a_nan && b_nan) || (a_inf && b_inf && ((a > 0) == (b > 0)));
            if (!allow_special_ref || !same) {
                ++v.bad;
                v.worst = i;
                v.max_abs = std::numeric_limits<double>::infinity();
            }
            continue;
        }
        if (b_nan || b_inf) { // 参考有限但结果非有限：一律失败
            ++v.bad;
            v.worst = i;
            v.max_abs = std::numeric_limits<double>::infinity();
            continue;
        }
        const double d = std::fabs(a - b);
        const double tol = atol + rtol * std::fabs(a);
        const double r = d / (std::fabs(a) > 1e-12 ? std::fabs(a) : 1e-12);
        if (d > tol) {
            ++v.bad;
        }
        if (d > v.max_abs) {
            v.max_abs = d;
            v.worst = i;
        }
        if (r > v.max_rel) {
            v.max_rel = r;
        }
    }
    return v;
}

bool Compare(const std::string &name, const std::vector<float> &ref, const std::vector<float> &got, double atol,
             double rtol, bool allow_special_ref = false) {
    const Verdict v = Evaluate(ref, got, atol, rtol, allow_special_ref);
    const bool ok = v.empty();
    if (!ok) {
        g_failed = true;
    }
    std::printf("%-42s %-5s n=%-6zu max_abs=%-11.3e max_rel=%-11.3e bad=%-3zu worst[%zu] ref=%.7g got=%.7g\n",
                name.c_str(), ok ? "PASS" : "FAIL", ref.size(), v.max_abs, v.max_rel, v.bad, v.worst,
                ref.empty() ? 0.0f : ref[v.worst], got.empty() ? 0.0f : got[v.worst]);
    return ok;
}

// 用解析期望值直接断言（不依赖 CPU 实现，R06）
bool CompareAnalytic(const std::string &name, const std::vector<double> &expect, const std::vector<float> &got,
                     double atol, double rtol) {
    std::vector<float> ref(expect.size());
    for (size_t i = 0; i < expect.size(); ++i) { ref[i] = static_cast<float>(expect[i]); }
    return Compare(name, ref, got, atol, rtol, /*allow_special_ref=*/true);
}

// ---------------------------------------------------------------------------
// 纯 host：比较器自检（R02/R06），不初始化设备
// ---------------------------------------------------------------------------
int RunComparatorSelftest() {
    struct Case {
        const char *name;
        std::vector<float> ref;
        std::vector<float> got;
        bool allow_special;
        bool expect_pass;
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const std::vector<Case> cases = {
        {"finite equal", {1.0f}, {1.0f}, false, true},
        {"finite within tol", {1.0f}, {1.0f + 5e-5f}, false, true},
        {"finite beyond tol", {1.0f}, {1.01f}, false, false},
        {"ref +Inf got 0", {inf}, {0.0f}, false, false},
        {"ref -Inf got 0", {-inf}, {0.0f}, false, false},
        {"ref +Inf got -Inf", {inf}, {-inf}, false, false},
        {"ref +Inf got +Inf allowed", {inf}, {inf}, true, true},
        {"ref +Inf got +Inf not allowed", {inf}, {inf}, false, false},
        {"ref NaN got NaN not allowed", {nan_v}, {nan_v}, false, false},
        {"ref NaN got NaN allowed", {nan_v}, {nan_v}, true, true},
        {"ref NaN got 1 allowed", {nan_v}, {1.0f}, true, false},
        {"ref finite got NaN", {1.0f}, {nan_v}, true, false},
        {"ref finite got Inf", {1.0f}, {inf}, true, false},
        {"shape mismatch", {1.0f, 2.0f}, {1.0f}, false, false},
        {"tolerance boundary just inside", {2.0f}, {2.0f + 1e-5f + 1e-4f * 2.0f * 0.99f}, false, true},
        {"tolerance boundary just outside", {2.0f}, {2.0f + 1e-5f + 1e-4f * 2.0f * 1.5f}, false, false},
    };
    int failures = 0;
    for (const auto &c : cases) {
        const Verdict v = Evaluate(c.ref, c.got, 1e-5, 1e-4, c.allow_special);
        const bool pass = v.empty();
        const bool ok = (pass == c.expect_pass);
        if (!ok) {
            ++failures;
        }
        std::printf("%-38s expect=%-4s actual=%-4s %s\n", c.name, c.expect_pass ? "PASS" : "FAIL",
                    pass ? "PASS" : "FAIL", ok ? "ok" : "MISMATCH");
    }
    std::printf("comparator-selftest: %s (%d mismatch)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 设备侧小算子用例
// ---------------------------------------------------------------------------
void ApplyInjection(std::vector<float> &mlu) {
    if (mlu.empty()) {
        return;
    }
    if (g_inject_finite) {
        mlu[0] += 1.0f;
    }
    if (g_inject_nan) {
        mlu[0] = std::numeric_limits<float>::quiet_NaN();
    }
}

void TestLinearForward(int64_t rows, int64_t in_f, int64_t out_f, bool transpose, bool with_bias) {
    auto x = RandVec(static_cast<size_t>(rows * in_f), 1.0f, 101);
    auto w = RandVec(static_cast<size_t>(in_f * out_f), 0.05f, 102);
    auto b = RandVec(static_cast<size_t>(out_f), 0.02f, 103);
    const std::vector<int64_t> x_dims{rows, in_f};
    const std::vector<int64_t> w_dims
        = transpose ? std::vector<int64_t>{out_f, in_f} : std::vector<int64_t>{in_f, out_f};
    auto run = [&](Device dev) {
        auto xt = MakeTensor(x_dims, x, dev);
        auto wt = MakeTensor(w_dims, w, dev);
        auto bt = with_bias ? MakeTensor({out_f}, b, dev) : nullptr;
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "LinearForward"}, xt,
                                                                                      wt, transpose, bt);
        return ToHost(out);
    };
    char name[128];
    std::snprintf(name, sizeof(name), "Linear fwd rows=%lld in=%lld out=%lld T=%d bias=%d",
                  static_cast<long long>(rows), static_cast<long long>(in_f), static_cast<long long>(out_f),
                  static_cast<int>(transpose), static_cast<int>(with_bias));
    auto cpu = run(Cpu());
    auto mlu = run(Mlu());
    ApplyInjection(mlu);
    Compare(name, cpu, mlu, 1e-5, 1e-4);
}

void TestLinearBackward(int64_t rows, int64_t in_f, int64_t out_f, bool transpose) {
    auto x = RandVec(static_cast<size_t>(rows * in_f), 1.0f, 201);
    auto w = RandVec(static_cast<size_t>(in_f * out_f), 0.05f, 202);
    auto g = RandVec(static_cast<size_t>(rows * out_f), 0.5f, 203);
    const std::vector<int64_t> x_dims{rows, in_f};
    const std::vector<int64_t> w_dims
        = transpose ? std::vector<int64_t>{out_f, in_f} : std::vector<int64_t>{in_f, out_f};
    const std::vector<int64_t> g_dims{rows, out_f};
    auto run = [&](Device dev) {
        auto wt = MakeTensor(w_dims, w, dev);
        auto gt = MakeTensor(g_dims, g, dev);
        auto xt = MakeTensor(x_dims, x, dev);
        auto dx = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "LinearBackwardInput"}, wt, gt, transpose, in_f, out_f, x_dims);
        auto dw = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "LinearBackwardWeight"}, xt, gt, transpose, in_f, out_f);
        auto db = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "LinearBackwardBias"},
                                                                                     gt, out_f);
        return std::make_tuple(ToHost(dx), ToHost(dw), ToHost(db));
    };
    auto [cpu_dx, cpu_dw, cpu_db] = run(Cpu());
    auto [mlu_dx, mlu_dw, mlu_db] = run(Mlu());
    ApplyInjection(mlu_dx);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "rows=%lld in=%lld out=%lld T=%d", static_cast<long long>(rows),
                  static_cast<long long>(in_f), static_cast<long long>(out_f), static_cast<int>(transpose));
    Compare(std::string("Linear bwd dx ") + tag, cpu_dx, mlu_dx, 1e-5, 1e-4);
    Compare(std::string("Linear bwd dw ") + tag, cpu_dw, mlu_dw, 1e-5, 1e-4);
    Compare(std::string("Linear bwd db ") + tag, cpu_db, mlu_db, 1e-5, 1e-4);
}

// Pow：解析期望（R06）——0^0.5=0, 0^0=1, (-2)^9=-512, (-2)^0.5=NaN, 2^0.5=sqrt2,
// (-2)^3=-8, 0.5^-1=2, base=3/exp=2 => 9
void TestPowAnalytic() {
    struct Case {
        float input;
        float exponent;
        bool base_is_scalar;
        double expect;
        const char *desc;
    };
    const Case cases[] = {
        {0.0f, 0.5f, false, 0.0, "0^0.5"},
        {0.0f, 0.0f, false, 1.0, "0^0"},
        {-2.0f, 9.0f, false, -512.0, "(-2)^9"},
        {-2.0f, 0.5f, false, std::numeric_limits<double>::quiet_NaN(), "(-2)^0.5"},
        {2.0f, 0.5f, false, std::sqrt(2.0), "2^0.5"},
        {-2.0f, 3.0f, false, -8.0, "(-2)^3"},
        {0.5f, -1.0f, false, 2.0, "0.5^-1"},
        {2.0f, 3.0f, true, 9.0, "base=3 exp=2 -> 9"}, // tensor=2 是指数，scalar=3 是底数
    };
    for (const auto &c : cases) {
        auto run = [&](Device dev) {
            auto xt = MakeTensor({1}, {c.input}, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
                {dev.type(), "PowForward"}, xt, c.exponent, c.base_is_scalar);
            return ToHost(out);
        };
        auto cpu = run(Cpu());
        auto got = run(Mlu());
        char name[96];
        std::snprintf(name, sizeof(name), "Pow %s", c.desc);
        // 解析期望与 CPU 参考双保险：先按解析值断言，再与 CPU 对拍
        CompareAnalytic(std::string(name) + " [analytic]", {c.expect}, got, 1e-6, 1e-5);
        Compare(std::string(name) + " [vs cpu]", cpu, got, 1e-6, 1e-5, /*allow_special_ref=*/true);
    }

    // scalar_is_base 反向：y=base^x, dy/dx = base^x*ln(base)；base=3, x=2, grad=2 -> 2*9*ln3
    {
        const float base = 3.0f, xval = 2.0f, grad = 2.0f;
        const double expect = static_cast<double>(grad) * std::pow(base, xval) * std::log(base);
        auto run = [&](Device dev) {
            auto gt = MakeTensor({1}, {grad}, dev);
            auto xt = MakeTensor({1}, {xval}, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowBackward"},
                                                                                          gt, xt, base, true);
            return ToHost(out);
        };
        CompareAnalytic("Pow bwd scalar_is_base=3 x=2 grad=2", {expect}, run(Mlu()), 1e-5, 1e-5);
    }

    // Pow(x,3) 前向/反向，输入显式含 0 与负值
    {
        const std::vector<float> xs = {0.0f, -2.0f, 0.5f, 3.0f};
        const std::vector<float> gs = {1.0f, 2.0f, -1.0f, 0.5f};
        std::vector<double> exp_f(4), exp_b(4);
        for (size_t i = 0; i < xs.size(); ++i) {
            const double x = xs[i];
            exp_f[i] = x * x * x;
            exp_b[i] = static_cast<double>(gs[i]) * 3.0 * x * x; // x=0 处为 0
        }
        auto run_fwd = [&](Device dev) {
            auto xt = MakeTensor({4}, xs, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowForward"},
                                                                                          xt, 3.0f, false);
            return ToHost(out);
        };
        auto run_bwd = [&](Device dev) {
            auto gt = MakeTensor({4}, gs, dev);
            auto xt = MakeTensor({4}, xs, dev);
            auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "PowBackward"},
                                                                                          gt, xt, 3.0f, false);
            return ToHost(out);
        };
        CompareAnalytic("Pow(x,3) fwd incl x=0", exp_f, run_fwd(Mlu()), 1e-6, 1e-5);
        CompareAnalytic("Pow(x,3) bwd incl x=0", exp_b, run_bwd(Mlu()), 1e-6, 1e-5);
    }
}

void TestCrossEntropy(int64_t bs, int64_t classes) {
    auto logits = RandVec(static_cast<size_t>(bs * classes), 1.0f, 401);
    std::vector<int64_t> target(static_cast<size_t>(bs));
    for (int64_t i = 0; i < bs; ++i) { target[i] = i % classes; }
    auto run = [&](Device dev) {
        auto lt = MakeTensor({bs, classes}, logits, dev);
        auto tt = MakeTensorI64({bs}, target, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {dev.type(), "CrossEntropyForward"}, lt, tt);
        return ToHost(out);
    };
    Compare("CrossEntropy fwd (bs=4,cls=8)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestLayerNorm(int64_t rows, int64_t dim) {
    auto x = RandVec(static_cast<size_t>(rows * dim), 1.0f, 11);
    auto w = RandVec(static_cast<size_t>(dim), 0.1f, 12);
    auto b = RandVec(static_cast<size_t>(dim), 0.1f, 13);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({rows, 1, dim}, x, dev);
        auto wt = MakeTensor({dim}, w, dev);
        auto bt = MakeTensor({dim}, b, dev);
        auto res = infini_train::Dispatcher::Instance()
                       .Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
                           {dev.type(), "LayerNormForward"}, xt, wt, bt, 1e-5f);
        return ToHost(std::get<0>(res));
    };
    Compare("LayerNorm", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestSoftmax(int64_t rows, int64_t cols) {
    auto x = RandVec(static_cast<size_t>(rows * cols), 1.0f, 21);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({rows, cols}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "SoftmaxForward"},
                                                                                      xt, int64_t{-1});
        return ToHost(out);
    };
    Compare("Softmax(dim=-1)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestAddBroadcast(int64_t rows, int64_t cols) {
    auto a = RandVec(static_cast<size_t>(rows * cols), 1.0f, 31);
    auto b = RandVec(static_cast<size_t>(cols), 0.5f, 32);
    auto run = [&](Device dev) {
        auto at = MakeTensor({rows, cols}, a, dev);
        auto bt = MakeTensor({cols}, b, dev);
        auto out
            = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "AddForward"}, at, bt);
        return ToHost(out);
    };
    Compare("Add(broadcast)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

void TestMulScalar(int64_t n, float s) {
    auto x = RandVec(static_cast<size_t>(n), 1.0f, 41);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({n}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "MulScalarForward"},
                                                                                      xt, s);
        return ToHost(out);
    };
    auto cpu = run(Cpu());
    auto mlu = run(Mlu());
    ApplyInjection(mlu);
    Compare("MulScalar", cpu, mlu, 1e-5, 1e-4);
}

void TestTranspose(int64_t b, int64_t h, int64_t t, int64_t d) {
    auto x = RandVec(static_cast<size_t>(b * h * t * d), 1.0f, 51);
    auto run = [&](Device dev) {
        auto xt = MakeTensor({b, h, t, d}, x, dev);
        auto out = infini_train::Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({dev.type(), "TransposeForward"},
                                                                                      xt, int64_t{1}, int64_t{2});
        return ToHost(out);
    };
    Compare("Transpose(1,2)", run(Cpu()), run(Mlu()), 1e-5, 1e-4);
}

// ---------------------------------------------------------------------------
// 探针用例（R05）：只分配 + 拷贝，不发 kernel；由脚本配合 MLU_DEBUG_CHECK=1 断言日志
// ---------------------------------------------------------------------------
template <typename Bits>
std::shared_ptr<Tensor> MakeRawTensor(const std::vector<int64_t> &dims, const std::vector<Bits> &bits, DataType dt,
                                      Device dev) {
    auto t = std::make_shared<Tensor>(dims, dt, dev);
    infini_train::core::GetDeviceGuardImpl(dev.type())
        ->Memcpy(t->DataPtr(), bits.data(), bits.size() * sizeof(Bits), infini_train::core::MemcpyKind::kH2D);
    return t;
}

int RunProbeCase(const std::string &name) {
    using infini_train::core::mlu::DebugCheck;
    if (name == "fp32_special") {
        // [1, NaN, +Inf, -Inf, 0] -> non_finite=3, 有限 min=0 max=1
        auto t = MakeTensor({5},
                            {1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(), 0.0f},
                            Mlu());
        DebugCheck<float>(t, "probe_fp32_special");
    } else if (name == "first_nan") {
        // 首元素 NaN：[NaN, 1, 2] -> non_finite=1, min=1 max=2（验证首个有限值初始化）
        auto t = MakeTensor({3}, {std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f}, Mlu());
        DebugCheck<float>(t, "probe_first_nan");
    } else if (name == "all_nonfinite") {
        auto t = MakeTensor({3},
                            {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()},
                            Mlu());
        DebugCheck<float>(t, "probe_all_nonfinite");
    } else if (name == "bf16_bits") {
        // 0x7c00 是合法有限 bf16（指数 0xf8），0x7f80 才是 +Inf，0x3f80=1.0，0x8000=-0
        std::vector<uint16_t> bits = {0x7c00, 0x7f80, 0x3f80, 0x8000};
        auto t = MakeRawTensor<uint16_t>({4}, bits, DataType::kBFLOAT16, Mlu());
        DebugCheck<uint16_t>(t, "probe_bf16_bits");
    } else if (name == "fp16_bits") {
        // 0x7c00 是 +Inf（fp16 指数 0x1f），0x3c00=1.0，0xfc00=-Inf
        std::vector<uint16_t> bits = {0x7c00, 0x3c00, 0xfc00};
        auto t = MakeRawTensor<uint16_t>({3}, bits, DataType::kFLOAT16, Mlu());
        DebugCheck<uint16_t>(t, "probe_fp16_bits");
    } else if (name == "int_wide") {
        // 宽整数范围（R04）：UINT64_MAX 与 INT64_MIN/MAX 不得显示成 -1 或溢出
        std::vector<uint64_t> u = {0, 18446744073709551615ULL};
        auto tu = MakeRawTensor<uint64_t>({2}, u, DataType::kUINT64, Mlu());
        DebugCheck<uint64_t>(tu, "probe_uint64");
        std::vector<int64_t> s = {INT64_MIN, INT64_MAX};
        auto ts = MakeRawTensor<int64_t>({2}, s, DataType::kINT64, Mlu());
        DebugCheck<int64_t>(ts, "probe_int64");
    } else {
        std::printf("unknown probe case: %s\n", name.c_str());
        return 2;
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    // 模式解析必须在 backend 注册/设备初始化之前（001-P2 v2 要求）
    std::string probe_case;
    std::string status_test;
    bool comparator_selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--comparator-selftest") {
            comparator_selftest = true;
        } else if (arg == "--inject-finite") {
            g_inject_finite = true;
        } else if (arg == "--inject-nan") {
            g_inject_nan = true;
        } else if (arg == "--debug-status-test" && i + 1 < argc) {
            status_test = argv[++i];
        } else if (arg == "--probe-case" && i + 1 < argc) {
            probe_case = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            g_device = std::atoi(argv[++i]);
        } else {
            std::printf("unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    if (comparator_selftest) {
        return RunComparatorSelftest();
    }

    // S03：受控失败路径测试。只调用既有 CheckSyncResult 的伪返回码，不把任何 queue 交给 CNRT，
    // 也不触发真实设备故障。父脚本按"唯一标识 + 预期终止 + 无后续标记"判定。
    if (!status_test.empty()) {
        // S03: controlled failure-path test. Calls the existing CheckSyncResult with a fabricated
        // return code only; no queue is handed to CNRT and no real device fault is provoked.
        if (status_test == "success") {
            std::cout << "STATUS_TEST_BEGIN success" << std::endl;
            infini_train::core::mlu::CheckSyncResult(cnrtSuccess, "debug-status-test-success");
            std::cout << "STATUS_TEST_END success (continued as expected)" << std::endl;
            return 0;
        }
        if (status_test == "failure") {
            std::cout << "STATUS_TEST_BEGIN failure" << std::endl;
            std::cout << "STATUS_TEST_ABORT_MARKER fake-cnrt-failure-632015" << std::endl;
            std::fflush(stdout);
            infini_train::core::mlu::CheckSyncResult(static_cast<cnrtRet_t>(632015), "debug-status-test-failure");
            std::cout << "STATUS_TEST_AFTER_FATAL should-not-appear" << std::endl;
            return 0;
        }
        std::cout << "unknown --debug-status-test value: " << status_test << std::endl;
        return 2;
    }

    infini_train::backends::mlu::RegisterBackend();
    std::printf("op_test: device=%d inject_finite=%d inject_nan=%d\n", g_device, static_cast<int>(g_inject_finite),
                static_cast<int>(g_inject_nan));

    if (!probe_case.empty()) {
        return RunProbeCase(probe_case);
    }

    for (bool transpose : {true, false}) {
        TestLinearForward(3, 5, 7, transpose, true);
        TestLinearForward(3, 5, 7, transpose, false);
        TestLinearBackward(3, 5, 7, transpose);
    }
    TestPowAnalytic();
    TestCrossEntropy(4, 8);
    TestLayerNorm(8, 16);
    TestSoftmax(8, 16);
    TestAddBroadcast(8, 16);
    TestMulScalar(64, 0.5f);
    TestTranspose(1, 2, 4, 8);

    std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0; // 注入模式下同样如实返回非 0，由外层脚本检查预期退出码
}
