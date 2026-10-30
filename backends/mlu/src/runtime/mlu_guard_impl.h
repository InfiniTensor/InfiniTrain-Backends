#pragma once

// 对应 master 上 src/core/runtime/cuda/cuda_guard_impl.h。
// 注意：master 的 PrivateUse1 机制下不再新增 DeviceType::kMLU 枚举，
// MLU 后端统一占用 DeviceType::kPrivateUse1。

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

namespace infini_train::core {
class Stream;
class BlasHandle;
} // namespace infini_train::core

namespace infini_train::core::mlu {

class MluGuardImpl final : public DeviceGuardImpl {
public:
    static void InitSingleStream(Device device);

    static void InitSingleHandle(Device device);

    MluGuardImpl();

    // device
    Device GetDevice() const override;

    void SetDevice(Device device) const override;

    int DeviceCount() const override;

    Device::DeviceType Type() const override;

    // stream
    Stream *GetStream(Device device) const override;

    // The pinned InfiniTrain revision (c90338c) does not declare
    // DeviceGuardImpl::ExchangeStream, so this backend does not override it. The
    // previous implementation was never called; if a later framework revision
    // brings the interface back, implement it here and mark it override again.

    Stream *CreateStream(Device device) const override;

    Stream *CreateStreamWithPriority(Device device, int priority) const override;

    void DestroyStream(Stream *stream) const override;

    void GetStreamPriorityRange(int *low, int *high) const override;

    // event
    void EventCreate(Event **event) const override;

    void EventCreateWithFlags(Event **event, EventFlag flags) const override;

    void EventDestroy(Event *event) const override;

    void EventRecord(Event *event, Stream *stream) const override;

    void StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const override;

    RuntimeStatus EventSynchronize(Event *event) const override;

    RuntimeStatus EventQuery(Event *event) const override;

    float EventElapsedTime(Event *start_event, Event *stop_event) const override;

    // sync
    void SynchronizeDevice(Device device) const override;
    void SynchronizeStream(Stream *stream) const override;

    // blas
    BlasHandle *GetBlasHandle(Device device) const override;

    // memory
    void Malloc(void **dev_ptr, size_t size) override;

    void MallocAsync(void **dev_ptr, size_t size, Stream *stream) override;

    void Free(void *dev_ptr) override;

    void FreeAsync(void *dev_ptr, Stream *stream) override;

    void Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) override;

    void MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) override;

    void ResetMemPoolHighWatermarks(Device device) const override;

    std::pair<size_t, size_t> GetMemPoolPeakMB(Device device) const override;
};

} // namespace infini_train::core::mlu
