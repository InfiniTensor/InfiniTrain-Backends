#pragma once

#include <cnnl.h>
#include <cnrt.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::mlu {

// Maps a framework EventFlag onto CNRT notifier creation flags.
unsigned int ToCnrtNotifierFlags(EventFlag flags);

class MluEvent final : public Event {
public:
    explicit MluEvent(EventFlag flags = EventFlag::kDefault);
    ~MluEvent() override;

    cnrtNotifier_t mlu_event() const;

private:
    cnrtNotifier_t event_ = nullptr;
};

class MluStream final : public Stream {
public:
    MluStream();
    explicit MluStream(int priority);

    // NOTE:
    // The wrapper destructor avoids CNRT calls because global destruction may
    // outlive the runtime. Explicitly owned streams must be released through
    // DeviceGuardImpl::DestroyStream while the runtime is alive.
    ~MluStream() override;

    cnrtQueue_t mlu_queue() const;

private:
    cnrtQueue_t queue_ = nullptr;
};

class MluBlasHandle final : public BlasHandle {
public:
    explicit MluBlasHandle(Stream *stream);

    // NOTE:
    // Mirror MluStream: leaked intentionally; released only when the runtime
    // is guaranteed to still be alive.
    ~MluBlasHandle() override;

    cnnlHandle_t cnnl_handle() const;

private:
    cnnlHandle_t cnnl_handle_ = nullptr;
};

} // namespace infini_train::core::mlu
