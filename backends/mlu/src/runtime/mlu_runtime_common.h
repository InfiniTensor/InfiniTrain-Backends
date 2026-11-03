#pragma once

// 对应 master 上 src/core/runtime/cuda/cuda_runtime_common.h。
// 概念映射：HIP/CUDA stream -> CNRT queue，event -> CNRT notifier，
// cublas handle -> CNNL handle。

#include <cstdint>

#ifdef USE_CNNL
#include <cnnl.h>
#endif
#include <cnrt.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::mlu {

class MluEvent final : public Event {
public:
    explicit MluEvent(EventFlag flags = EventFlag::kDefault);
    ~MluEvent() override;

    cnrtNotifier_t mlu_notifier() const;

private:
    cnrtNotifier_t notifier_ = nullptr;
};

class MluStream : public Stream {
public:
    MluStream();
    explicit MluStream(int priority);

    // NOTE: 与 CUDA 实现同理——全局 stream 故意不销毁。进程退出时静态对象的
    // 析构可能发生在 CNRT runtime 销毁之后，此时调 cnrtQueueDestroy 会崩溃，
    // 因此“忘掉”销毁。
    ~MluStream() override;

    cnrtQueue_t mlu_queue() const;

private:
    cnrtQueue_t queue_ = nullptr;
};

// 调试用：环境变量 MLU_DEBUG_SYNC=1 时在每个 kernel 启动后同步队列并打日志，
// 用来定位设备侧崩溃（CNRT 的 CN_INVOKE_ERROR 不报是哪个 kernel）。
void DebugSyncQueue(cnrtQueue_t queue, const char *kernel_name);

#ifdef USE_CNNL
class MluBlasHandle : public BlasHandle {
public:
    explicit MluBlasHandle(Stream *stream);

    // NOTE: 同上，故意不销毁（cnnlDestroy 在退出期不安全）。
    ~MluBlasHandle() override;

    cnnlHandle_t cnnl_handle() const;

private:
    cnnlHandle_t cnnl_handle_;
};
#endif // USE_CNNL

} // namespace infini_train::core::mlu
