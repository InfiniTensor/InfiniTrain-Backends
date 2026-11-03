#include "runtime/mlu_runtime_common.h"

#include "runtime/mlu_debug.h"

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "common/common_mlu.h"

namespace infini_train::core::mlu {

MluEvent::MluEvent(EventFlag /*flags*/) {
    // CNRT notifier 没有创建 flag；EventFlag 的 timing/blocking 语义由
    // wait/query 调用点承载。
    CNRT_CHECK(cnrtNotifierCreate(&notifier_));
}

MluEvent::~MluEvent() {
    if (notifier_ != nullptr) {
        CNRT_CHECK(cnrtNotifierDestroy(notifier_));
    }
}

cnrtNotifier_t MluEvent::mlu_notifier() const { return notifier_; }

MluStream::MluStream() { CNRT_CHECK(cnrtQueueCreate(&queue_)); }

MluStream::MluStream(int priority) { CNRT_CHECK(cnrtQueueCreateWithPriority(&queue_, 0, priority)); }

MluStream::~MluStream() {
    // Do nothing.
}

cnrtQueue_t MluStream::mlu_queue() const { return queue_; }

#ifdef USE_CNNL
MluBlasHandle::MluBlasHandle(Stream *stream) {
    CNNL_CHECK(cnnlCreate(&cnnl_handle_));
    // 已知坑 #1：handle 必须绑定当前 queue，否则异步执行顺序错乱。
    CNNL_CHECK(cnnlSetQueue(cnnl_handle_, dynamic_cast<MluStream *>(stream)->mlu_queue()));
}

MluBlasHandle::~MluBlasHandle() {
    // Do nothing.
}

cnnlHandle_t MluBlasHandle::cnnl_handle() const { return cnnl_handle_; }
#endif // USE_CNNL

bool DebugFlagEnabled(const char *env_name) {
    const char *raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    const std::string v(raw);
    if (v == "0" || v == "false" || v == "FALSE" || v == "off") {
        return false;
    }
    if (v == "1" || v == "true" || v == "TRUE" || v == "on") {
        return true;
    }
    LOG(WARNING) << "env " << env_name << "=" << v << " is not a recognised flag value; treated as disabled";
    return false;
}

bool CheckSyncResult(cnrtRet_t ret, const char *kernel_name) {
    if (ret == cnrtSuccess) {
        return true;
    }
    // 探针开启时属于排障场景：SDK 失败后队列已不可信，必须停止，避免后续错误掩盖第一失败点。
    int dev_id = -1;
    cnrtGetDevice(&dev_id);
    LOG(FATAL) << "MLU debug: kernel '" << kernel_name << "' failed (cnrtQueueSync ret=" << static_cast<int>(ret)
               << ", device=" << dev_id << ")";
    return false;
}

void DebugSyncQueue(cnrtQueue_t queue, const char *kernel_name) {
    if (!DebugFlagEnabled("MLU_DEBUG_SYNC")) {
        return;
    }
    CheckSyncResult(cnrtQueueSync(queue), kernel_name);
    LOG(INFO) << "MLU kernel synced: " << kernel_name;
}

void DebugCheckTensor(const void *dev_ptr, int64_t num_elements, int dtype, const char *name) {
    if (!DebugFlagEnabled("MLU_DEBUG_CHECK")) {
        return;
    }
    if (num_elements <= 0) {
        return;
    }

    // 探针内部也走当前流，否则读到的是流水线上的旧数据（历史踩过的坑）。
    if (cnrtSyncDevice() != cnrtSuccess) {
        LOG(FATAL) << "MLU debug: cnrtSyncDevice failed before probing " << name;
    }

    const int64_t kMaxSample = 262144;
    const int64_t n = num_elements < kMaxSample ? num_elements : kMaxSample;

    // 按真实 dtype 解码，禁止用元素字节数猜类型
    enum : int { kBOOL = 0, kU8, kI8, kU16, kI16, kU32, kI32, kU64, kI64, kBF16, kF16, kF32, kF64 };
    int elem_size = 0;
    switch (dtype) {
    case kBOOL:
    case kU8:
    case kI8:
        elem_size = 1;
        break;
    case kU16:
    case kI16:
    case kBF16:
    case kF16:
        elem_size = 2;
        break;
    case kU32:
    case kI32:
    case kF32:
        elem_size = 4;
        break;
    case kU64:
    case kI64:
    case kF64:
        elem_size = 8;
        break;
    default:
        LOG(WARNING) << "[check] " << name << ": unsupported dtype id " << dtype << ", skipped";
        return;
    }

    std::vector<char> host(static_cast<size_t>(n) * static_cast<size_t>(elem_size));
    if (cnrtMemcpy(host.data(), const_cast<void *>(dev_ptr), host.size(), cnrtMemcpyDevToHost) != cnrtSuccess) {
        LOG(FATAL) << "MLU debug: D2H copy failed while probing " << name;
    }

    int64_t non_finite = 0;
    int64_t finite = 0;
    double vmin = 0.0, vmax = 0.0;
    bool is_float = true;

    auto push = [&](double v, bool finite_v) {
        if (!finite_v) {
            ++non_finite;
            return;
        }
        if (finite == 0) {
            vmin = vmax = v;
        }
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
        ++finite;
    };

    switch (dtype) {
    case kF32: {
        const auto *p = reinterpret_cast<const float *>(host.data());
        for (int64_t i = 0; i < n; ++i) { push(static_cast<double>(p[i]), std::isfinite(p[i])); }
        break;
    }
    case kBF16: {
        const auto *p = reinterpret_cast<const uint16_t *>(host.data());
        for (int64_t i = 0; i < n; ++i) {
            const float v = infini_train::detail::Bf16BitsToFloat(p[i]);
            // bf16 的 inf/nan 位型：指数位（bit7..14）全 1
            const bool finite_v = ((p[i] >> 7) & 0xFF) != 0xFF;
            push(static_cast<double>(v), finite_v);
        }
        break;
    }
    case kF16: {
        const auto *p = reinterpret_cast<const uint16_t *>(host.data());
        for (int64_t i = 0; i < n; ++i) {
            const float v = infini_train::detail::Fp16BitsToFloat(p[i]);
            // fp16 的 inf/nan 位型：指数位（bit10..14）全 1
            const bool finite_v = ((p[i] >> 10) & 0x1F) != 0x1F;
            push(static_cast<double>(v), finite_v);
        }
        break;
    }
    case kF64: {
        const auto *p = reinterpret_cast<const double *>(host.data());
        for (int64_t i = 0; i < n; ++i) { push(p[i], std::isfinite(p[i])); }
        break;
    }
    case kBOOL: {
        const auto *p = reinterpret_cast<const uint8_t *>(host.data());
        for (int64_t i = 0; i < n; ++i) { push(static_cast<double>(p[i]), true); }
        break;
    }
    default: { // 整型：无 NaN 概念；按原生位宽统计，不经过 double，避免 UINT64_MAX/INT64_MAX 失真
        is_float = false;
        const char *p = host.data();
        bool is_unsigned = (dtype == kU8 || dtype == kU16 || dtype == kU32 || dtype == kU64 || dtype == kBOOL);
        if (is_unsigned && (dtype == kU64 || dtype == kU32 || dtype == kU16 || dtype == kU8 || dtype == kBOOL)) {
            uint64_t vmin_u = 0, vmax_u = 0;
            for (int64_t i = 0; i < n; ++i) {
                uint64_t v = 0;
                switch (dtype) {
                case kBOOL:
                case kU8:
                    v = reinterpret_cast<const uint8_t *>(p)[i];
                    break;
                case kU16:
                    v = reinterpret_cast<const uint16_t *>(p)[i];
                    break;
                case kU32:
                    v = reinterpret_cast<const uint32_t *>(p)[i];
                    break;
                default:
                    v = reinterpret_cast<const uint64_t *>(p)[i];
                    break;
                }
                if (i == 0 || v < vmin_u) {
                    vmin_u = v;
                }
                if (i == 0 || v > vmax_u) {
                    vmax_u = v;
                }
            }
            std::ostringstream os;
            os << "[check] " << name << " dtype=" << dtype << " n=" << num_elements << " sampled=" << n
               << (n < num_elements ? "(prefix)" : "(full)") << " non_finite=0 uint_min=" << vmin_u
               << " uint_max=" << vmax_u;
            LOG(INFO) << os.str();
            return;
        }
        int64_t vmin_s = 0, vmax_s = 0;
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = 0;
            switch (dtype) {
            case kI8:
                v = reinterpret_cast<const int8_t *>(p)[i];
                break;
            case kI16:
                v = reinterpret_cast<const int16_t *>(p)[i];
                break;
            case kI32:
                v = reinterpret_cast<const int32_t *>(p)[i];
                break;
            default:
                v = reinterpret_cast<const int64_t *>(p)[i];
                break;
            }
            if (i == 0 || v < vmin_s) {
                vmin_s = v;
            }
            if (i == 0 || v > vmax_s) {
                vmax_s = v;
            }
        }
        std::ostringstream os2;
        os2 << "[check] " << name << " dtype=" << dtype << " n=" << num_elements << " sampled=" << n
            << (n < num_elements ? "(prefix)" : "(full)") << " non_finite=0 int_min=" << vmin_s
            << " int_max=" << vmax_s;
        LOG(INFO) << os2.str();
        return;
    }
    }

    std::ostringstream os;
    os << "[check] " << name << " dtype=" << dtype << " n=" << num_elements << " sampled=" << n
       << (n < num_elements ? "(prefix)" : "(full)") << " non_finite=" << non_finite;
    if (finite == 0) {
        os << " min=NA max=NA (no finite element)";
    } else if (is_float) {
        os << " min=" << vmin << " max=" << vmax;
    } else {
        os << " min=" << static_cast<int64_t>(vmin) << " max=" << static_cast<int64_t>(vmax);
    }
    LOG(INFO) << os.str();
}

} // namespace infini_train::core::mlu
