#include "runtime/maca_guard_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/core/runtime/runtime_common.h"
#include "infini_train/include/device.h"

#include "common/common_maca.h"
#include "runtime/maca_runtime_common.h"

namespace infini_train::core::maca {
namespace {
// Read /proc/self/cmdline and return --tensor_parallel value, or 1 if absent
// or unparseable. This does not depend on gflags and runs before mcInit.
int ReadTensorParallelFromCmdline() {
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    if (!in) {
        return 1;
    }

    std::vector<std::string> args;
    std::string current;
    char c;
    while (in.get(c)) {
        if (c == '\0') {
            if (!current.empty()) {
                args.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        args.push_back(std::move(current));
    }

    constexpr char kTensorParallelFlag[] = "--tensor_parallel";
    constexpr char kTensorParallelPrefix[] = "--tensor_parallel=";
    for (size_t i = 0; i < args.size(); ++i) {
        std::string value;
        if (args[i].rfind(kTensorParallelPrefix, 0) == 0) {
            value = args[i].substr(sizeof(kTensorParallelPrefix) - 1);
        } else if (args[i] == kTensorParallelFlag && i + 1 < args.size()) {
            value = args[i + 1];
        } else {
            continue;
        }

        try {
            return std::stoi(value);
        } catch (...) { return 1; }
    }
    return 1;
}

static std::vector<std::unique_ptr<MacaStream>> g_maca_streams;
static std::vector<std::unique_ptr<MacaBlasHandle>> g_maca_blas_handles;
static std::vector<std::unique_ptr<std::once_flag>> g_device_stream_flags;
static std::vector<std::unique_ptr<std::once_flag>> g_device_handle_flags;

// Serialize host-side MemcpyAsync across threads. On MACA, concurrent
// mcMemcpyAsync from multiple threads during init-time bursts
// (Module::To uploads, Adam state fills, ...) races with the runtime's
// auto P2P peer-mapping and produces "readonly page" faults or
// mcErrorInvalidValue. The lock is held only for the brief window of the
// API call itself; actual GPU work remains async on the caller's stream.
static std::mutex g_memcpy_mutex;

inline void CheckMacaDevice(Device device) {
    CHECK(device.type() == Device::DeviceType::kPrivateUse1) << std::format(
        "MacaGuardImpl expects MACA device, but got type={} index={}", static_cast<int>(device.type()), device.index());
    const int idx = device.index();
    CHECK(idx >= 0 && static_cast<size_t>(idx) < g_maca_streams.size())
        << std::format("MACA device index {} out of cache range [0, {}).", idx, g_maca_streams.size());
}

inline mcEvent_t GetMacaEvent(Event *event) {
    auto *maca_event = dynamic_cast<MacaEvent *>(event);
    CHECK_NOTNULL(maca_event);
    return maca_event->maca_event();
}

inline mcStream_t GetMacaStream(Stream *stream) {
    auto *maca_stream = dynamic_cast<MacaStream *>(stream);
    CHECK_NOTNULL(maca_stream);
    return maca_stream->maca_stream();
}
} // namespace

void MacaGuardImpl::InitSingleStream(Device device) {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    g_maca_streams[device.index()] = std::make_unique<MacaStream>();

    MACA_CHECK(mcSetDevice(current_device));
}

void MacaGuardImpl::InitSingleHandle(Device device) {
    CheckMacaDevice(device);

    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    std::call_once(*g_device_stream_flags.at(device.index()), InitSingleStream, device);

    g_maca_blas_handles[device.index()] = std::make_unique<MacaBlasHandle>(g_maca_streams[device.index()].get());

    MACA_CHECK(mcSetDevice(current_device));
}

void MacaGuardImpl::Initialize() {
    std::call_once(initialize_flag_, [] {
        // FIXME(cx): Stop deriving runtime policy from argv and mutating process-wide
        // environment here. Pass MACA runtime/communication policy through explicit
        // provider or launcher configuration instead.
        // Apply provider runtime policy immediately before mcInit. Users may
        // override it in the process environment before first device use.
        setenv("MACA_LAUNCH_BLOCKING", "1", 0);
        if (ReadTensorParallelFromCmdline() > 1) {
            setenv("MCCL_P2P_DISABLE", "1", 0);
        }
        MACA_CHECK(mcInit(0));

        int device_count = 0;
        MACA_CHECK(mcGetDeviceCount(&device_count));
        CHECK_GT(device_count, 0) << "No MACA devices are available.";
        CHECK_LE(static_cast<size_t>(device_count), static_cast<size_t>(std::numeric_limits<int8_t>::max()) + 1)
            << "MACA device count exceeds InfiniTrain's Device index range.";

        g_maca_streams.resize(device_count);
        g_maca_blas_handles.resize(device_count);
        g_device_stream_flags.reserve(device_count);
        g_device_handle_flags.reserve(device_count);
        for (int i = 0; i < device_count; ++i) {
            g_device_stream_flags.push_back(std::make_unique<std::once_flag>());
            g_device_handle_flags.push_back(std::make_unique<std::once_flag>());
        }
    });
}

// device
Device MacaGuardImpl::GetDevice() const {
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    return Device(Device::DeviceType::kPrivateUse1, current_device);
}

void MacaGuardImpl::SetDevice(Device device) const {
    CheckMacaDevice(device);
    MACA_CHECK(mcSetDevice(device.index()));
}

int MacaGuardImpl::DeviceCount() const { return static_cast<int>(g_maca_streams.size()); }

Device::DeviceType MacaGuardImpl::Type() const { return Device::DeviceType::kPrivateUse1; }

// stream
Stream *MacaGuardImpl::GetStream(Device device) const {
    CheckMacaDevice(device);
    std::call_once(*g_device_stream_flags.at(device.index()), InitSingleStream, device);
    return g_maca_streams.at(device.index()).get();
}

Stream *MacaGuardImpl::CreateStream(Device device) const {
    CheckMacaDevice(device);
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    Stream *stream = new MacaStream();

    MACA_CHECK(mcSetDevice(current_device));
    return stream;
}

Stream *MacaGuardImpl::CreateStreamWithPriority(Device device, int priority) const {
    CheckMacaDevice(device);
    int current_device = -1;
    MACA_CHECK(mcGetDevice(&current_device));
    MACA_CHECK(mcSetDevice(device.index()));

    Stream *stream = new MacaStream(priority);

    MACA_CHECK(mcSetDevice(current_device));
    return stream;
}

void MacaGuardImpl::DestroyStream(Stream *stream) const {
    if (stream == nullptr) {
        return;
    }
    auto *maca_stream = dynamic_cast<MacaStream *>(stream);
    CHECK_NOTNULL(maca_stream);
    MACA_CHECK(mcStreamDestroy(maca_stream->maca_stream()));
    delete maca_stream;
}

void MacaGuardImpl::GetStreamPriorityRange(int *low, int *high) const {
    MACA_CHECK(mcDeviceGetStreamPriorityRange(low, high));
}

// event
void MacaGuardImpl::EventCreate(Event **event) const { *event = new MacaEvent(); }

void MacaGuardImpl::EventCreateWithFlags(Event **event, EventFlag flags) const { *event = new MacaEvent(flags); }

void MacaGuardImpl::EventDestroy(Event *event) const {
    if (event == nullptr) {
        return;
    }
    delete event;
}

void MacaGuardImpl::EventRecord(Event *event, Stream *stream) const {
    auto maca_event = GetMacaEvent(event);
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcEventRecord(maca_event, maca_stream));
}

void MacaGuardImpl::StreamWaitEvent(Stream *stream, Event *event, uint32_t flags) const {
    auto maca_event = GetMacaEvent(event);
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcStreamWaitEvent(maca_stream, maca_event, flags));
}

RuntimeStatus MacaGuardImpl::EventSynchronize(Event *event) const {
    auto maca_event = GetMacaEvent(event);
    mcError_t status = mcEventSynchronize(maca_event);
    if (status == mcSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == mcErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MacaGuardImpl::EventSynchronize failed: " << mcGetErrorString(status);
    return RuntimeStatus::kError;
}

RuntimeStatus MacaGuardImpl::EventQuery(Event *event) const {
    auto maca_event = GetMacaEvent(event);
    mcError_t status = mcEventQuery(maca_event);
    if (status == mcSuccess) {
        return RuntimeStatus::kSuccess;
    }
    if (status == mcErrorNotReady) {
        return RuntimeStatus::kNotReady;
    }
    LOG(ERROR) << "MacaGuardImpl::EventQuery failed: " << mcGetErrorString(status);
    return RuntimeStatus::kError;
}

float MacaGuardImpl::EventElapsedTime(Event *start_event, Event *stop_event) const {
    auto start_maca_event = GetMacaEvent(start_event);
    auto stop_maca_event = GetMacaEvent(stop_event);
    float elapsed_ms = 0.0f;
    MACA_CHECK(mcEventElapsedTime(&elapsed_ms, start_maca_event, stop_maca_event));
    return elapsed_ms;
}

// sync
void MacaGuardImpl::SynchronizeDevice(Device device) const {
    auto original_device = GetDevice();
    SetDevice(device);

    MACA_CHECK(mcDeviceSynchronize());

    SetDevice(original_device);
}

void MacaGuardImpl::SynchronizeStream(Stream *stream) const {
    auto maca_stream = GetMacaStream(stream);
    MACA_CHECK(mcStreamSynchronize(maca_stream));
}

// blas
BlasHandle *MacaGuardImpl::GetBlasHandle(Device device) const {
    CheckMacaDevice(device);
    std::call_once(*g_device_handle_flags.at(device.index()), InitSingleHandle, device);
    return g_maca_blas_handles.at(device.index()).get();
}

// memory
void MacaGuardImpl::Malloc(void **dev_ptr, size_t size) { MACA_CHECK(mcMalloc(dev_ptr, size)); }

void MacaGuardImpl::MallocAsync(void **dev_ptr, size_t size, Stream *stream) {
    // NOTE(dcj): mcMallocAsync with a per-stream mempool gives a big speedup
    // (~2x on gpt2 DDP steady-state) vs synchronous mcMalloc, but under
    // multi-thread DDP init bursts (e.g. llama3 1B with nthread=8 uploading
    // hundreds of param tensors) it races with MACA's auto P2P peer-mapping
    // and produces mcErrorInvalidValue on subsequent mcMemcpyAsync, or
    // "readonly page" faults -- no amount of mutex/stream-sync serialization
    // around the alloc call suppresses this. Keep the synchronous path for
    // correctness.
    // auto maca_stream = GetMacaStream(stream);
    // MACA_CHECK(mcMallocAsync(dev_ptr, size, maca_stream));
    (void)stream;
    Malloc(dev_ptr, size);
}

void MacaGuardImpl::Free(void *dev_ptr) { MACA_CHECK(mcFree(dev_ptr)); }

void MacaGuardImpl::FreeAsync(void *dev_ptr, Stream *stream) {
    // auto maca_stream = GetMacaStream(stream);
    // MACA_CHECK(mcFreeAsync(dev_ptr, maca_stream));
    (void)stream;
    Free(dev_ptr);
}

void MacaGuardImpl::Memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
    if (kind == MemcpyKind::kH2D) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyHostToDevice));
    } else if (kind == MemcpyKind::kD2H) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyDeviceToHost));
    } else if (kind == MemcpyKind::kD2D) {
        MACA_CHECK(mcMemcpy(dst, src, count, mcMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << std::format("MacaGuardImpl::Memcpy got invalid MemcpyKind={}", MemcpyKindToString(kind));
    }
}

void MacaGuardImpl::MemcpyAsync(void *dst, const void *src, size_t count, MemcpyKind kind, Stream *stream) {
    std::lock_guard<std::mutex> lock(g_memcpy_mutex);
    auto maca_stream = GetMacaStream(stream);

    switch (kind) {
    case MemcpyKind::kH2D:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyHostToDevice, maca_stream));
        break;
    case MemcpyKind::kD2H:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyDeviceToHost, maca_stream));
        break;
    case MemcpyKind::kD2D:
        MACA_CHECK(mcMemcpyAsync(dst, src, count, mcMemcpyDeviceToDevice, maca_stream));
        break;
    default:
        LOG(FATAL) << std::format("MacaGuardImpl::MemcpyAsync got invalid MemcpyKind={}", MemcpyKindToString(kind));
    }
}

void MacaGuardImpl::ResetMemPoolHighWatermarks(Device device) const {
    // MetaX SDK support for the mempool high-watermark attributes is not
    // confirmed. Keep this a no-op, matching feat/muxi_device_registry.
    (void)device;
}

std::pair<size_t, size_t> MacaGuardImpl::GetMemPoolPeakMB(Device device) const {
    (void)device;
    return std::make_pair<size_t, size_t>(0, 0);
}

void RegisterMacaRuntime() { INFINI_TRAIN_REGISTER_DEVICE_GUARD_IMPL(Device::DeviceType::kPrivateUse1, MacaGuardImpl) }

} // namespace infini_train::core::maca
