// 对应 master 上 src/core/runtime/cuda/cuda_guard_impl.cc，CUDA API 全部换成
// CNRT/CNNL 对应物（对照表见工作区根目录 DCU_MLU_API映射速查表.md）。
//
// 已对照服务器 /usr/local/neuware/include/cnrt.h（CNRT 7.7.3）核实：
//   notifier 系列为 cnrtNotifierCreate/Destroy/Place/Wait/Query/ElapsedTime；
//   queue 支持 cnrtQueueCreateWithPriority；memcpy 方向枚举是 cnrtMemTransDir_t；
//   cnrtMemcpyAsync(dst, src, bytes, queue, dir) 参数顺序与 HIP 相反；
//   无 cnrtInit（自动初始化）、无 cnrtMallocAsync/cnrtFreeAsync。

#include "runtime/mlu_guard_impl.h"

#include <array>
#include <memory>
#include <mutex>
#include <vector>

#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "common/common_mlu.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::core::mlu {
namespace {
constexpr int kMaxGpus = 8;
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

static std::array<std::unique_ptr<MluStream>, kMaxGpus> mlu_queues;
static thread_local std::array<Stream *, kMaxGpus> current_queues = {};
#ifdef USE_CNNL
// 每个线程在每个设备上独占一个 handle；handle 在线程退出后留在池里复用，
// 因为 MluBlasHandle 退出期故意不销毁底层资源。
struct BlasHandlePool {
    std::mutex mutex;
    std::array<std::vector<std::shared_ptr<MluBlasHandle>>, kMaxGpus> handles;
};
static BlasHandlePool mlu_blas_handle_pool;
static thread_local std::array<std::shared_ptr<MluBlasHandle>, kMaxGpus> mlu_blas_handles;
#endif // USE_CNNL

static std::array<std::once_flag, kMaxGpus> device_stream_flags;

inline void CheckMluDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kPrivateUse1)
        << "MluGuardImpl expects PrivateUse1 (MLU) device, but got type=" << static_cast<int>(device.type())
        << " index=" << static_cast<int>(device.index());
    const int idx = device.index();
    CHECK(idx >= 0 && idx < kMaxGpus) << "MLU device index " << idx << " out of cache range [0, " << kMaxGpus << ").";
}

inline cnrtNotifier_t GetMluNotifier(Event *event) {
    auto *mlu_event = dynamic_cast<MluEvent *>(event);
    CHECK_NOTNULL(mlu_event);
    return mlu_event->mlu_notifier();
}

inline cnrtQueue_t GetMluQueue(Stream *stream) {
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    return mlu_stream->mlu_queue();
}

inline cnrtMemTransDir_t ToCnrtMemcpyKind(MemcpyKind kind) {
    switch (kind) {
    case MemcpyKind::kH2D:
        return cnrtMemcpyHostToDev;
    case MemcpyKind::kD2H:
        return cnrtMemcpyDevToHost;
    case MemcpyKind::kD2D:
        return cnrtMemcpyDevToDev;
    default:
        LOG(FATAL) << "Invalid MemcpyKind=" << MemcpyKindToString(kind);
    }
    return cnrtMemcpyDevToDev; // unreachable
}
} // namespace

void MluGuardImpl::InitSingleStream(Device device) {
    CheckMluDevice(device);

    int current_device = -1;
    CNRT_CHECK(cnrtGetDevice(&current_device));
    CNRT_CHECK(cnrtSetDevice(device.index()));

    mlu_queues[device.index()] = std::make_unique<MluStream>();
    current_queues[device.index()] = mlu_queues[device.index()].get();

    CNRT_CHECK(cnrtSetDevice(current_device));
}

void MluGuardImpl::InitSingleHandle(Device device) {
#ifdef USE_CNNL
    CheckMluDevice(device);

    auto &handle = mlu_blas_handles.at(device.index());
    if (handle) {
        return;
    }
    std::lock_guard<std::mutex> lock(mlu_blas_handle_pool.mutex);
    auto &pooled_handles = mlu_blas_handle_pool.handles.at(device.index());
    for (const auto &pooled_handle : pooled_handles) {
        // use_count() == 1 表示只有池持有，即当前没有线程占用该 handle。
        if (pooled_handle.use_count() == 1) {
            handle = pooled_handle;
            return;
        }
    }

    int current_device = -1;
    CNRT_CHECK(cnrtGetDevice(&current_device));
    CNRT_CHECK(cnrtSetDevice(device.index()));

    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);

    handle = std::make_shared<MluBlasHandle>(mlu_queues[device.index()].get());
    pooled_handles.push_back(handle);

    CNRT_CHECK(cnrtSetDevice(current_device));
#endif // USE_CNNL
}

MluGuardImpl::MluGuardImpl() {}

// device
Device MluGuardImpl::GetDevice() const {
    int current_device = -1;
    CNRT_CHECK(cnrtGetDevice(&current_device));
    return Device(Device::DeviceType::kPrivateUse1, current_device);
}

void MluGuardImpl::SetDevice(Device device) const {
    CheckMluDevice(device);
    CNRT_CHECK(cnrtSetDevice(device.index()));
}

int MluGuardImpl::DeviceCount() const {
    unsigned int device_count = 0;
    CNRT_CHECK(cnrtGetDeviceCount(&device_count));
    return static_cast<int>(device_count);
}

Device::DeviceType MluGuardImpl::Type() const { return Device::DeviceType::kPrivateUse1; }

// stream
Stream *MluGuardImpl::GetStream(Device device) const {
    CheckMluDevice(device);
    std::call_once(device_stream_flags.at(device.index()), InitSingleStream, device);
    if (current_queues.at(device.index()) == nullptr) {
        current_queues.at(device.index()) = mlu_queues.at(device.index()).get();
    }
    return current_queues.at(device.index());
}

Stream *MluGuardImpl::CreateStream(Device device) const {
    CheckMluDevice(device);
    int current_device = -1;
    CNRT_CHECK(cnrtGetDevice(&current_device));
    CNRT_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream();

    CNRT_CHECK(cnrtSetDevice(current_device));
    return stream;
}

Stream *MluGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckMluDevice(device);
    int current_device = -1;
    CNRT_CHECK(cnrtGetDevice(&current_device));
    CNRT_CHECK(cnrtSetDevice(device.index()));

    Stream *stream = new MluStream(priority);

    CNRT_CHECK(cnrtSetDevice(current_device));
    return stream;
}

void MluGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *mlu_stream = dynamic_cast<MluStream *>(stream);
    CHECK_NOTNULL(mlu_stream);
    delete mlu_stream;
}

void MluGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    // CNRT queue 无优先级概念。
    *low = 0;
    *high = 0;
}

// event
void MluGuardImpl::EventCreate(Event **event) const { *event = new MluEvent(); }

void MluGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new MluEvent(flags); }

void MluGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    delete event;
}

void MluGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto notifier = GetMluNotifier(event);
    auto queue = GetMluQueue(stream);
    CNRT_CHECK(cnrtPlaceNotifier(notifier, queue));
}

void MluGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto notifier = GetMluNotifier(event);
    auto queue = GetMluQueue(stream);
    // 注意参数顺序：CNRT 是 notifier 在前、queue 在后，与 HIP 相反。
    CNRT_CHECK(cnrtQueueWaitNotifier(notifier, queue, flags));
}

RuntimeStatus MluGuardImpl::EventSynchronize(Event *event) const {
    auto notifier = GetMluNotifier(event);
    cnrtRet_t status = cnrtWaitNotifier(notifier);
    if (status == cnrtSuccess) {
        return RuntimeStatus::kSuccess;
    }
    LOG(ERROR) << "MluGuardImpl::EventSynchronize failed: " << cnrtGetErrorStr(status);
    return RuntimeStatus::kError;
}

RuntimeStatus MluGuardImpl::EventQuery(Event *event) const {
    auto notifier = GetMluNotifier(event);
    cnrtRet_t status = cnrtQueryNotifier(notifier);
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
    auto start_notifier = GetMluNotifier(start_event);
    auto stop_notifier = GetMluNotifier(stop_event);
    float elapsed_ms = 0.0f;
    CNRT_CHECK(cnrtNotifierElapsedTime(start_notifier, stop_notifier, &elapsed_ms));
    return elapsed_ms;
}

// sync
void MluGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    CNRT_CHECK(cnrtSyncDevice());

    SetDevice(original_device);
}

void MluGuardImpl::SynchronizeStream(Stream *stream) const {
    auto queue = GetMluQueue(stream);
    CNRT_CHECK(cnrtQueueSync(queue));
}

// blas
BlasHandle *MluGuardImpl::GetBlasHandle(Device device) const {
#ifdef USE_CNNL
    CheckMluDevice(device);
    if (!mlu_blas_handles.at(device.index())) {
        InitSingleHandle(device);
    }
    auto *handle = mlu_blas_handles.at(device.index()).get();
    // 已知坑 #1：每次取用都重新绑定当前 queue，保证与上层调度 stream 一致。
    CNNL_CHECK(cnnlSetQueue(handle->cnnl_handle(), GetMluQueue(GetStream(device))));
    return handle;
#else
    LOG(FATAL) << "MluGuardImpl::GetBlasHandle requires CNNL (build with -DUSE_CNNL=ON "
                  "after installing the CNNL package)";
    return nullptr; // unreachable
#endif
}

// memory
void MluGuardImpl::Malloc(void **dev_ptr, size_t size) { CNRT_CHECK(cnrtMalloc(dev_ptr, size)); }

void MluGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // 保底策略（AGENTS.md §五.6）：CNRT 7.7.3 无异步分配接口，用同步 cnrtMalloc。
    CNRT_CHECK(cnrtMalloc(dev_ptr, size));
}

void MluGuardImpl::Free(void *dev_ptr) { CNRT_CHECK(cnrtFree(dev_ptr)); }

void MluGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    // 保底策略：先等 queue 排空再同步释放，避免释放后队列中还有访问该地址的任务。
    auto queue = GetMluQueue(stream);
    CNRT_CHECK(cnrtQueueSync(queue));
    CNRT_CHECK(cnrtFree(dev_ptr));
}

void MluGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    // 关键：kernel 跑在设备当前 stream 上，而 cnrtMemcpy 只保证与默认流有序，
    // 直接用它会在 kernel 还没跑完时就把数据拷走（实测读到上一个张量的残留）。
    // 因此走当前 stream 的异步拷贝 + 同步。
    auto queue = GetMluQueue(GetStream(GetDevice()));
    if (queue != nullptr) {
        CNRT_CHECK(cnrtMemcpyAsync(dst, const_cast<void *>(src), count, queue, ToCnrtMemcpyKind(kind)));
        CNRT_CHECK(cnrtQueueSync(queue));
        return;
    }
    CNRT_CHECK(cnrtMemcpy(dst, const_cast<void *>(src), count, ToCnrtMemcpyKind(kind)));
}

void MluGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    auto queue = GetMluQueue(stream);
    // 注意参数顺序：CNRT 是 queue 在前、方向在最后，与 HIP 相反。
    CNRT_CHECK(cnrtMemcpyAsync(dst, const_cast<void *>(src), count, queue, ToCnrtMemcpyKind(kind)));
}

void MluGuardImpl::ResetMemPoolHighWatermarks(Device /*device*/) const {
    // CNRT 没有 mempool 高水位概念，无可重置项。
}

std::pair<size_t, size_t> MluGuardImpl::GetMemPoolPeakMB(Device device) const {
    // CNRT 只有全局空闲/总量查询（对当前设备）：used=reserved=total-free
    // （无独立显存池）。
    CheckMluDevice(device);
    const auto original_device = GetDevice();
    SetDevice(device);
    size_t free_bytes = 0, total_bytes = 0;
    CNRT_CHECK(cnrtMemGetInfo(&free_bytes, &total_bytes));
    SetDevice(original_device);
    const size_t used_mb = (total_bytes - free_bytes) / kBytesPerMB;
    return {used_mb, used_mb};
}

INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kPrivateUse1, MluGuardImpl)

} // namespace infini_train::core::mlu
