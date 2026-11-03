// 设备数学函数 / 启动方式自测。
//
// 默认套件（正确性）：单任务串行 + cnrtFuncTypeBlock 下的 exp/log/tanh/sqrt/rsqrt/pow
// 以及一次整块拷贝（identity），与 host 参考对拍；任何数值不匹配、非预期 NaN、同步失败
// 都使进程退出码非 0。
//
// 旧诊断模式（--legacy-union-diag）：保留历史 Union1 + 多任务索引的对照实验。
// 该组合已知会导致循环写 GDRAM 只有首尾落盘，因此**预期失败**；退出码照实反映失败，
// 只用于回归对照，不计入默认套件的通过数。
//
// 用法：./math_selftest [--device N] [--legacy-union-diag]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cnrt.h>

#define CHECK_CNRT(call)                                                                                               \
    do {                                                                                                               \
        const cnrtRet_t _r = (call);                                                                                   \
        if (_r != cnrtSuccess) {                                                                                       \
            std::printf("[FATAL] CNRT failed: %s -> %d\n", #call, static_cast<int>(_r));                               \
            std::exit(3);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {
enum MathOp : int { kExp = 0, kLog = 1, kTanh = 2, kSqrt = 3, kRsqrt = 4, kPow3 = 5, kIdentity = 6, kOpCount = 7 };

const char *OpName(int op) {
    switch (op) {
    case kExp:
        return "expf";
    case kLog:
        return "logf";
    case kTanh:
        return "tanhf";
    case kSqrt:
        return "sqrtf";
    case kRsqrt:
        return "rsqrt(1/sqrtf)";
    case kPow3:
        return "powf(x,3)";
    case kIdentity:
        return "identity(x)";
    default:
        return "?";
    }
}

float HostRef(int op, float x) {
    switch (op) {
    case kExp:
        return std::exp(x);
    case kLog:
        return std::log(x);
    case kTanh:
        return std::tanh(x);
    case kSqrt:
        return std::sqrt(x);
    case kRsqrt:
        return 1.0f / std::sqrt(x);
    case kPow3:
        return x * x * x;
    case kIdentity:
        return x;
    default:
        return 0.0f;
    }
}

// 统一非有限判定（S04）：ref 非有限时要求同名同号（域内 NaN/Inf 可预期）；
// ref 有限时 got 必须有限，再比较 atol+rtol*|ref|。
bool Matches(double a, double b, double atol, double rtol) {
    const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
    const bool a_inf = std::isinf(a), b_inf = std::isinf(b);
    if (a_nan || a_inf) {
        return (a_nan && b_nan) || (a_inf && b_inf && ((a > 0) == (b > 0)));
    }
    if (b_nan || b_inf) {
        return false;
    }
    return std::fabs(a - b) <= atol + rtol * std::fabs(a);
}

// 纯 host 反例：不启动设备，验证有限/同号 Inf/异号 Inf/预期 NaN 规则
int RunCompareSelftest() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    struct C {
        const char *name;
        double ref;
        double got;
        bool expect;
    };
    const C cases[] = {
        {"finite equal", 1.0, 1.0, true},          {"finite beyond tol", 1.0, 1.01, false},
        {"ref +Inf got finite", inf, 0.0, false},  {"ref -Inf got finite", -inf, 0.0, false},
        {"ref +Inf got -Inf", inf, -inf, false},   {"ref +Inf got +Inf", inf, inf, true},
        {"ref NaN got NaN", nan_v, nan_v, true},   {"ref NaN got finite", nan_v, 1.0, false},
        {"ref finite got NaN", 1.0, nan_v, false}, {"ref finite got Inf", 1.0, inf, false},
    };
    int bad = 0;
    for (const auto &c : cases) {
        const bool got = Matches(c.ref, c.got, 1e-6, 1e-5);
        const bool ok = (got == c.expect);
        if (!ok) {
            ++bad;
        }
        std::cout << c.name << " expect=" << (c.expect ? "PASS" : "FAIL") << " actual=" << (got ? "PASS" : "FAIL")
                  << " " << (ok ? "ok" : "MISMATCH") << std::endl;
    }
    std::cout << "compare-selftest: " << (bad == 0 ? "PASS" : "FAIL") << " (" << bad << " mismatch)" << std::endl;
    return bad == 0 ? 0 : 1;
}

} // namespace

void MathSelftestLaunch(const void *in, void *out, int64_t n, int op, cnrtQueue_t queue);
void Probe3Launch(const void *in, void *out, int64_t n, cnrtQueue_t queue);
void ProbeLaunch(void *out, int64_t n, cnrtQueue_t queue);
void Probe2Launch(const void *in, void *out, int64_t n, int mode, cnrtQueue_t queue);
void Probe4Launch(void *out, cnrtQueue_t queue);
void Probe5Launch(const void *in, void *out, int64_t n, cnrtQueue_t queue);

int main(int argc, char **argv) {
    int device = 0;
    bool legacy = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--legacy-union-diag") {
            legacy = true;
        } else if (a == "--compare-selftest") {
            continue; // 在参数解析后、设备初始化前处理
        } else if (a == "--device" && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else {
            std::printf("unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    constexpr int64_t kN = 1024;
    std::vector<float> x(kN);
    for (int64_t i = 0; i < kN; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kN);
        x[i] = -3.0f + 6.0f * t;
        if (i % 64 == 0) {
            x[i] = 0.5f;
        }
        if (i % 64 == 1) {
            x[i] = 1.0f;
        }
        if (i % 64 == 2) {
            x[i] = 2.0f;
        }
        if (i % 64 == 3) {
            x[i] = 8.5f;
        }
        if (i % 64 == 4) {
            x[i] = 0.001f;
        }
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--compare-selftest") {
            return RunCompareSelftest();
        }
    }

    CHECK_CNRT(cnrtSetDevice(device));

    void *dev_in = nullptr;
    void *dev_out = nullptr;
    CHECK_CNRT(cnrtMalloc(&dev_in, kN * sizeof(float)));
    CHECK_CNRT(cnrtMalloc(&dev_out, kN * sizeof(float)));
    CHECK_CNRT(cnrtMemcpy(dev_in, x.data(), kN * sizeof(float), cnrtMemcpyHostToDev));

    cnrtQueue_t queue = nullptr;
    CHECK_CNRT(cnrtQueueCreate(&queue));

    std::printf("math_selftest: device=%d mode=%s\n", device, legacy ? "legacy-union-diag" : "correctness(Block)");

    int failures = 0;

    if (legacy) {
        // 历史诊断：Union1 + 多任务索引，已知会丢写；退出码照实反映失败
        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        ProbeLaunch(dev_out, kN, queue);
        CHECK_CNRT(cnrtQueueSync(queue));
        std::vector<float> p(kN);
        CHECK_CNRT(cnrtMemcpy(p.data(), dev_out, kN * sizeof(float), cnrtMemcpyDevToHost));
        std::printf("legacy probe(const write): out[0]=%.1f out[1]=%.1f out[n-1]=%.1f\n", p[0], p[1], p[kN - 1]);

        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        Probe2Launch(dev_in, dev_out, kN, 2, queue);
        CHECK_CNRT(cnrtQueueSync(queue));
        CHECK_CNRT(cnrtMemcpy(p.data(), dev_out, kN * sizeof(float), cnrtMemcpyDevToHost));
        int64_t mismatch = 0;
        for (int64_t i = 0; i < kN; ++i) {
            if (p[i] != x[i]) {
                ++mismatch;
            }
        }
        std::printf("legacy probe2(read in[i]) mismatch=%lld/%lld (expect mismatch under Union1)\n",
                    static_cast<long long>(mismatch), static_cast<long long>(kN));
        if (mismatch != 0) {
            ++failures;
        }

        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        Probe4Launch(dev_out, queue);
        CHECK_CNRT(cnrtQueueSync(queue));
        CHECK_CNRT(cnrtMemcpy(p.data(), dev_out, kN * sizeof(float), cnrtMemcpyDevToHost));
        std::printf("legacy probe4(const write, 1 task) out[0]=%.1f out[5]=%.1f\n", p[0], p[5]);
        if (p[0] != 7.0f || p[5] != 8.0f) {
            ++failures;
        }

        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        Probe5Launch(dev_in, dev_out, 8, queue);
        CHECK_CNRT(cnrtQueueSync(queue));
        CHECK_CNRT(cnrtMemcpy(p.data(), dev_out, 8 * sizeof(float), cnrtMemcpyDevToHost));
        int64_t small_bad = 0;
        for (int64_t i = 0; i < 8; ++i) {
            if (p[i] != x[i]) {
                ++small_bad;
            }
        }
        std::printf("legacy probe5(loop n=8) mismatch=%lld/8\n", static_cast<long long>(small_bad));
        if (small_bad != 0) {
            ++failures;
        }

        std::printf("legacy mode result: %d failing probe(s)\n", failures);
        CHECK_CNRT(cnrtQueueDestroy(queue));
        cnrtFree(dev_in);
        cnrtFree(dev_out);
        return failures == 0 ? 0 : 1;
    }

    // 默认正确性套件：Block + 串行
    {
        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        Probe3Launch(dev_in, dev_out, kN, queue);
        if (cnrtQueueSync(queue) != cnrtSuccess) {
            // R03：队列已不可信，立即停止，不再向同一队列提交任何 kernel
            std::printf("[FAIL] identity copy: queue sync failed -> abort (no further launches)\n");
            cnrtQueueDestroy(queue);
            cnrtFree(dev_in);
            cnrtFree(dev_out);
            std::printf("RESULT: FAIL\n");
            return 3;
        } else {
            std::vector<float> got(kN);
            CHECK_CNRT(cnrtMemcpy(got.data(), dev_out, kN * sizeof(float), cnrtMemcpyDevToHost));
            int64_t bad = 0;
            for (int64_t i = 0; i < kN; ++i) {
                if (got[i] != x[i]) {
                    ++bad;
                }
            }
            std::printf("%-18s %s  mismatch=%lld/%lld\n", "identity (copy)", bad == 0 ? "PASS" : "FAIL",
                        static_cast<long long>(bad), static_cast<long long>(kN));
            if (bad != 0) {
                ++failures;
            }
        }
    }

    std::vector<float> got(kN);
    for (int op = 0; op < kOpCount; ++op) {
        CHECK_CNRT(cnrtMemset(dev_out, 0, kN * sizeof(float)));
        MathSelftestLaunch(dev_in, dev_out, kN, op, queue);
        if (cnrtQueueSync(queue) != cnrtSuccess) {
            std::printf("%-18s FAIL  queue sync failed (stop)\n", OpName(op));
            ++failures;
            break; // 队列已不可信，按方案停止
        }
        CHECK_CNRT(cnrtMemcpy(got.data(), dev_out, kN * sizeof(float), cnrtMemcpyDevToHost));

        double max_abs = 0.0, max_rel = 0.0;
        long long bad = 0;
        for (int64_t i = 0; i < kN; ++i) {
            const double a = static_cast<double>(HostRef(op, x[i]));
            const double b = static_cast<double>(got[i]);
            if (!Matches(a, b, 1e-6, 1e-5)) {
                ++bad;
            }
            if (std::isfinite(a) && std::isfinite(b)) {
                const double d = std::fabs(a - b);
                const double r = d / (std::fabs(a) > 1e-12 ? std::fabs(a) : 1e-12);
                if (d > max_abs) {
                    max_abs = d;
                }
                if (r > max_rel) {
                    max_rel = r;
                }
            }
        }
        const bool ok = bad == 0;
        if (!ok) {
            ++failures;
        }
        std::printf("%-18s %s  max_abs=%.3e max_rel=%.3e bad=%lld/%lld\n", OpName(op), ok ? "PASS" : "FAIL", max_abs,
                    max_rel, bad, static_cast<long long>(kN));
    }

    CHECK_CNRT(cnrtQueueDestroy(queue));
    cnrtFree(dev_in);
    cnrtFree(dev_out);
    std::printf("RESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
