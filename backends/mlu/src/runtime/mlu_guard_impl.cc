#include "runtime/mlu_guard_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "common/common_mlu.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::core::mlu {
namespace {
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

static std::vector<std::unique_ptr<MluStream>> g_mlu_streams;
static std::vector<std::unique_ptr<MluBlasHandle>> g_mlu_blas_handles;
static std::vector<std::unique_ptr<std::once_flag>> g_device_stream_flags;
static std::vector<std::unique_ptr<std::once_flag>> g_device_handle_flags;

struct AllocationRecord {
    int device_index = -1;
    size_t size = 0;
};

struct AllocationStats {
    size_t current = 0;
    size_t high = 0;
};

// CNRT allocations go straight to the driver with no caching pool, so the
// allocation tracker below is the only source of memory statistics.
static std::mutex g_allocation_mutex;
static std::unordered_map<void *, AllocationRecord> g_allocations;
static std::vector<AllocationStats> g_allocation_stats;

void TrackAllocation(void *ptr, int device_index, size_t size) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    g_allocations[ptr] = AllocationRecord{device_index, size};
    auto &stats = g_allocation_stats.at(device_index);
    stats.current += size;
    stats.high = std::max(stats.high, stats.current);
}

void UntrackAllocation(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) {
        return;
    }

    auto &stats = g_allocation_stats.at(it->second.device_index);
    CHECK_GE(stats.current, it->second.size) << "MLU allocation tracker underflow.";
    stats.current -= it->second.size;
    g_allocations.erase(it);
}

inline void CheckMluDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kPrivateUse1)
        << "MluGuardImpl expects MLU device, but got type=" << static_cast<int>(device.type())
        << " index=" << device.index();
    const int idx = device.index();
    CHECK(idx >= 0 && static_cast<size_t>(idx) < g_mlu_streams.size())
        << "MLU device index " << idx << " out of cache range [0, " << g_mlu_streams.size() << ").";
}

inline cnrtNotifier_t GetMluEvent(Event *event) {
    auto *mlu_event = dynamic_cast<MluEvent *>(event);
    CHECK_NOTNULL(mlu_event);
    return mlu_event->mlu_event();
}

inline cnrtQueue_t GetMluQueue(Stream *stream) {
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    return mlu_stream->mlu_queue();
}

cnrtMemcpyKind_t ToCnrtMemcpyKind(MemcpyKind kind, const char *context) {
    switch (kind) {
    case MemcpyKind::kH2D:
        return cnrtMemcpyHostToDev;
    case MemcpyKind::kD2H:
        return cnrtMemcpyDevToHost;
    case MemcpyKind::kD2D:
        return cnrtMemcpyDevToDev;
    default:
        LOG(FATAL) << context << " got invalid MemcpyKind=" << MemcpyKindToString(kind);
    }
}
} // namespace

void MluGuardImpl::InitSingleStream(Device device) {
    CheckMluDevice(device);

    int current_device = -1;
    MLU_CHECK(cnrtGetDevice(&current_device));
    MLU_CHECK(cnrtSetDevice(device.index()));

    g_mlu_streams[device.index()] = std::make_unique<MluStream>();

    MLU_CHECK(cnrtSetDevice(current_device));
}

void MluGuardImpl::InitSingleHandle(Device device) {
    CheckMluDevice(device);

    int current_device = -1;
    MLU_CHECK(cnrtGetDevice(&current_device));
    MLU_CHECK(cnrtSetDevice(device.index()));

    std::call_once(*g_device_stream_flags.at(device.index()), InitSingleStream, device);

    g_mlu_blas_handles[device.index()] = std::make_unique<MluBlasHandle>(g_mlu_streams[device.index()].get());

    MLU_CHECK(cnrtSetDevice(current_device));
}

void MluGuardImpl::Initialize() {
    std::call_once(initialize_flag_, [] {
        // CNRT 7.x initializes lazily on the first device API call; there is no
        // explicit global init entry point to invoke here.
        unsigned int device_count = 0;
        MLU_CHECK(cnrtGetDeviceCount(&device_count));
        CHECK_GT(static_cast<int>(device_count), 0) << "No MLU devices are available.";
        CHECK_LE(static_cast<size_t>(device_count), static_cast<size_t>(std::numeric_limits<int8_t>::max()) + 1)
            << "MLU device count exceeds InfiniTrain's Device index range.";

        g_mlu_streams.resize(device_count);
        g_mlu_blas_handles.resize(device_count);
        g_allocation_stats.resize(device_count);
        g_device_stream_flags.reserve(device_count);
        g_device_handle_flags.reserve(device_count);
        for (unsigned int i = 0; i < device_count; ++i) {
            g_device_stream_flags.push_back(std::make_unique<std::once_flag>());
            g_device_handle_flags.push_back(std::make_unique<std::once_flag>());
        }
    });
}

// device
Device MluGuardImpl::GetDevice() const {
    int current_device = -1;
    MLU_CHECK(cnrtGetDevice(&current_device));
    return Device(Device::DeviceType::kPrivateUse1, current_device);
}

void MluGuardImpl::SetDevice(Device device) const {
    CheckMluDevice(device);
    MLU_CHECK(cnrtSetDevice(device.index()));
}

int MluGuardImpl::DeviceCount() const { return static_cast<int>(g_mlu_streams.size()); }

Device::DeviceType MluGuardImpl::Type() const { return Device::DeviceType::kPrivateUse1; }

// stream
Stream *MluGuardImpl::GetStream(Device device) const {
    CheckMluDevice(device);
    std::call_once(*g_device_stream_flags.at(device.index()), InitSingleStream, device);
    return g_mlu_streams.at(device.index()).get();
}

Stream *MluGuardImpl::CreateStream(Device device) const {
    CheckMluDevice(device);
    int current_device = -1;
    MLU_CHECK(cnrtGetDevice(&current_device));
    MLU_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream();

    MLU_CHECK(cnrtSetDevice(current_device));
    return stream;
}

Stream *MluGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckMluDevice(device);
    int current_device = -1;
    MLU_CHECK(cnrtGetDevice(&current_device));
    MLU_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream(priority);

    MLU_CHECK(cnrtSetDevice(current_device));
    return stream;
}

void MluGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    MLU_CHECK(cnrtQueueDestroy(mlu_stream->mlu_queue()));
    delete mlu_stream;
}

void MluGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    // CNRT reports (min, max) accepted by cnrtQueueCreateWithPriority; pass the
    // range through in the framework's (low, high) convention.
    MLU_CHECK(cnrtDeviceGetQueuePriorityRange(low, high));
}

// event
void MluGuardImpl::EventCreate(Event **event) const { *event = new MluEvent(); }

void MluGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new MluEvent(flags); }

void MluGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    // The wrapped notifier is intentionally not destroyed here; see the
    // MluStream/MluEvent teardown notes.
    delete event;
}

void MluGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto mlu_event = GetMluEvent(event);
    auto mlu_queue = GetMluQueue(stream);
    MLU_CHECK(cnrtPlaceNotifier(mlu_event, mlu_queue));
}

void MluGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto mlu_event = GetMluEvent(event);
    auto mlu_queue = GetMluQueue(stream);
    MLU_CHECK(cnrtQueueWaitNotifier(mlu_event, mlu_queue, flags));
}

RuntimeStatus MluGuardImpl::EventSynchronize(Event *event) const {
    auto mlu_event = GetMluEvent(event);
    cnrtRet_t status = cnrtWaitNotifier(mlu_event);
    if (status == cnrtSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == cnrtErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MluGuardImpl::EventSynchronize failed: " << cnrtGetErrorStr(status);
    return RuntimeStatus::kError;
}

RuntimeStatus MluGuardImpl::EventQuery(Event *event) const {
    auto mlu_event = GetMluEvent(event);
    cnrtRet_t status = cnrtQueryNotifier(mlu_event);
    if (status == cnrtSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == cnrtErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MluGuardImpl::EventQuery failed: " << cnrtGetErrorStr(status);
    return RuntimeStatus::kError;
}

float MluGuardImpl::EventElapsedTime(Event *start_event, Event *stop_event) const {
    auto start_mlu_event = GetMluEvent(start_event);
    auto stop_mlu_event = GetMluEvent(stop_event);
    float elapsed_ms = 0.0f;
    MLU_CHECK(cnrtNotifierDuration(start_mlu_event, stop_mlu_event, &elapsed_ms));
    return elapsed_ms;
}

// sync
void MluGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    MLU_CHECK(cnrtSyncDevice());

    SetDevice(original_device);
}

void MluGuardImpl::SynchronizeStream(Stream *stream) const {
    auto mlu_queue = GetMluQueue(stream);
    MLU_CHECK(cnrtQueueSync(mlu_queue));
}

// blas
BlasHandle *MluGuardImpl::GetBlasHandle(Device device) const {
    CheckMluDevice(device);
    std::call_once(*g_device_handle_flags.at(device.index()), InitSingleHandle, device);
    return g_mlu_blas_handles.at(device.index()).get();
}

// memory
void MluGuardImpl::Malloc(void **dev_ptr, size_t size) {
    const cnrtRet_t status = cnrtMalloc(dev_ptr, size);
    if (status != cnrtSuccess) {
        int device_index = -1;
        size_t free_bytes = 0, total_bytes = 0;
        const cnrtRet_t device_status = cnrtGetDevice(&device_index);
        const cnrtRet_t memory_status = cnrtMemGetInfo(&free_bytes, &total_bytes);
        LOG(ERROR) << "MLU allocation failed: requested_bytes=" << size << " device=" << device_index
                   << " free_bytes=" << free_bytes << " total_bytes=" << total_bytes
                   << " device_query=" << cnrtGetErrorStr(device_status)
                   << " memory_query=" << cnrtGetErrorStr(memory_status);
    }
    MLU_CHECK(status);

    int device_index = -1;
    MLU_CHECK(cnrtGetDevice(&device_index));
    TrackAllocation(*dev_ptr, device_index, size);
}

void MluGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // CNRT 7.x has no asynchronous allocation API; allocate synchronously on
    // the allocation device. The stream argument is unused.
    (void)stream;
    Malloc(dev_ptr, size);
}

void MluGuardImpl::Free(void *dev_ptr) {
    UntrackAllocation(dev_ptr);
    MLU_CHECK(cnrtFree(dev_ptr));
}

void MluGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    if (dev_ptr == nullptr) {
        return;
    }
    // With synchronous allocation there may still be queued work touching this
    // block, so wait for the stream before releasing it.
    SynchronizeStream(stream);
    Free(dev_ptr);
}

void MluGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    const cnrtMemcpyKind_t cnrt_kind = ToCnrtMemcpyKind(kind, "MluGuardImpl::Memcpy");
    MLU_CHECK(cnrtMemcpy(dst, const_cast<void *>(src), count, cnrt_kind));
}

void MluGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    const cnrtMemcpyKind_t cnrt_kind = ToCnrtMemcpyKind(kind, "MluGuardImpl::MemcpyAsync");
    auto mlu_queue = GetMluQueue(stream);
    MLU_CHECK(cnrtMemcpyAsync(dst, const_cast<void *>(src), count, mlu_queue, cnrt_kind));
}

void MluGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    CheckMluDevice(device);

    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    auto &stats = g_allocation_stats.at(device.index());
    stats.high = stats.current;
}

std::pair<size_t, size_t> MluGuardImpl::GetMemPoolPeakMB(Device device) const {
    CheckMluDevice(device);

    size_t tracked_high = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        tracked_high = g_allocation_stats.at(device.index()).high;
    }

    const size_t peak_mb = tracked_high / kBytesPerMB;
    return {peak_mb, peak_mb};
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kPrivateUse1, MluGuardImpl)

} // namespace infini_train::core::mlu
