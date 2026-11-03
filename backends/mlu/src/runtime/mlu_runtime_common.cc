#include "runtime/mlu_runtime_common.h"

#include "common/common_mlu.h"

namespace infini_train::core::mlu {

unsigned int ToCnrtNotifierFlags(EventFlag flags) {
    switch (flags) {
    case EventFlag::kBlockingSync:
        return CNRT_NOTIFIER_SYNC_WAIT;
    case EventFlag::kDisableTiming:
        return CNRT_NOTIFIER_DISABLE_TIMING;
    case EventFlag::kInterprocess:
        LOG(WARNING) << "MLU notifiers do not support interprocess events; falling back to default flags.";
        return CNRT_NOTIFIER_DEFAULT;
    case EventFlag::kDefault:
        return CNRT_NOTIFIER_DEFAULT;
    }
    LOG(FATAL) << "Unsupported EventFlag: " << static_cast<uint32_t>(flags);
    return CNRT_NOTIFIER_DEFAULT;
}

MluEvent::MluEvent(EventFlag flags) { MLU_CHECK(cnrtNotifierCreateWithFlags(&event_, ToCnrtNotifierFlags(flags))); }

// See the class comment: no CNRT calls during teardown.
MluEvent::~MluEvent() = default;

cnrtNotifier_t MluEvent::mlu_event() const { return event_; }

MluStream::MluStream() { MLU_CHECK(cnrtQueueCreate(&queue_)); }

MluStream::MluStream(int priority) { MLU_CHECK(cnrtQueueCreateWithPriority(&queue_, 0, priority)); }

// See the class comment: no CNRT calls during teardown.
MluStream::~MluStream() = default;

cnrtQueue_t MluStream::mlu_queue() const { return queue_; }

MluBlasHandle::MluBlasHandle(Stream *stream) {
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);

    CNNL_CHECK(cnnlCreate(&cnnl_handle_));
    CNNL_CHECK(cnnlSetQueue(cnnl_handle_, mlu_stream->mlu_queue()));
}

// See the class comment: no CNNL calls during teardown.
MluBlasHandle::~MluBlasHandle() = default;

cnnlHandle_t MluBlasHandle::cnnl_handle() const { return cnnl_handle_; }

} // namespace infini_train::core::mlu
