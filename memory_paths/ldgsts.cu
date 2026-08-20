// Standalone LDGSTS (cp.async) global-memory->SMEM effective-copy
// microbenchmark: the LDGSTS arm of the "LDGSTS versus TMA" experiment.
//
// Frozen experimental contract: method=ldgsts, PTX cp.async.cg.shared.global, 16-byte
// vectorization, 128 threads/CTA, a maximum active residency of 1 CTA/SM
// (enforced by an oversized dynamic-shared-memory reservation and checked
// with the occupancy API), grid = SM count, stages in {2,4,8}, bytes-in-flight/SM in
// {16,32,64} KiB. The nine (stages, bytes-in-flight) combinations are
// reached through two kernel templates (validate, benchmark) instantiated
// via runtime dispatch — the kernel body itself is written once.
//
//   stage_bytes                  = bytes_in_flight_per_sm / stages
//   copies_per_thread_per_stage  = stage_bytes / (128 threads * 16 bytes)
//   bytes_in_flight_per_sm       = stages * stage_bytes
//
// Logical layout (shared with the TMA arm): 128 BF16-width
// (2-byte) elements per row = 256 bytes/row, tile_height = stage_bytes/256.
// LDGSTS itself copies each tile as one contiguous linear region; the row
// layout is metadata for CSV/TMA compatibility, not something the copy loop
// depends on.
//
// Conceptual references only (no code copied): the PTX ISA chapter on
// cp.async/cp.async.commit_group/cp.async.wait_group, and the CUDA
// Programming Guide's asynchronous-copy section. This file is an
// independent implementation written against that documentation, not a
// derivative of any third-party benchmark source.
//
// Exit codes: 0 = success (or --help), 1 = validation/CUDA/self-test
// failure, 2 = command-line usage error.
//
// This binary never selects a GPU, never reads any environment variable
// except EXPECTED_GPU_UUID (set by scripts/run_gpu.sh), and requires
// exactly one visible device at compute capability 10.3.

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// Frozen contract constants.
// ---------------------------------------------------------------------------
constexpr int kThreadsPerCta = 128;
constexpr int kVectorBytes = 16;
constexpr int kTargetMaxActiveCtasPerSm = 1;
constexpr int kTileWidthElements = 128;  // BF16-width (2-byte) elements/row
constexpr int kTileWidthBytes = 256;     // 128 * 2 bytes
constexpr int64_t kSmemAlignmentBytes = 128;
constexpr uint64_t kPatternSalt = 0xD1B54A32D192ED03ULL;
constexpr const char* kSchemaVersion = "1";
constexpr const char* kMethodName = "ldgsts";

// Self-test working set: a fixed, small multiple of the common-multiple
// unit (sm_count * 32 KiB), independent of any user-supplied working set.
// 8x gives 256 KiB/CTA, i.e. 8..128 tiles/CTA across all nine
// specializations — enough to exercise pipeline fill/steady-state/drain.
constexpr int64_t kSelfTestCommonMultiples = 8;

int g_cleanup_failures = 0;

enum class RunStatus {
    kOk,
    kMismatch,
    kCudaError,
};

const char* run_status_name(RunStatus status) {
    switch (status) {
        case RunStatus::kOk: return "PASS";
        case RunStatus::kMismatch: return "MISMATCH";
        case RunStatus::kCudaError: return "CUDA_ERROR";
    }
    return "UNKNOWN";
}

[[noreturn]] void fail(const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "ldgsts: ERROR: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// RAII CUDA resource wrappers. A CUDA_CHECK failure inside run_specialization
// returns a distinct CUDA-error status through local wrappers, so every
// allocation made so far is still released during ordinary stack unwind.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "ldgsts: cuda_error=%s detail=\"%s\" at %s:%d\n", \
                         cudaGetErrorName(err_), cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                               \
            return RunStatus::kCudaError;                                   \
        }                                                                   \
    } while (0)

template <typename T>
class DeviceBuffer {
 public:
    explicit DeviceBuffer(const char* label) : label_(label) {}
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                             "ldgsts: cleanup_error=%s detail=\"%s\" buffer=%s\n",
                             cudaGetErrorName(err), cudaGetErrorString(err), label_);
                ++g_cleanup_failures;
            }
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    cudaError_t allocate(size_t count) { return cudaMalloc(&ptr_, count * sizeof(T)); }
    T* get() const { return ptr_; }

 private:
    const char* label_;
    T* ptr_ = nullptr;
};

class CudaEvent {
 public:
    CudaEvent() = default;
    ~CudaEvent() {
        if (created_) {
            cudaError_t err = cudaEventDestroy(ev_);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                             "ldgsts: cleanup_error=%s detail=\"%s\" resource=cuda_event\n",
                             cudaGetErrorName(err), cudaGetErrorString(err));
                ++g_cleanup_failures;
            }
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaError_t create() {
        cudaError_t err = cudaEventCreate(&ev_);
        if (err == cudaSuccess) created_ = true;
        return err;
    }
    cudaEvent_t get() const { return ev_; }

 private:
    cudaEvent_t ev_{};
    bool created_ = false;
};

// ---------------------------------------------------------------------------
// The nine frozen specializations, derived from the formulas (not
// hand-copied from the table) so the formulas are the single source of
// truth. The (stages, bif_kib) pairs enumerated below are exactly the
// cartesian product of the frozen "Allowed stages" and "Bytes in flight per
// SM" sets from the contract.
// ---------------------------------------------------------------------------
struct Specialization {
    int stages = 0;
    int bif_kib = 0;
    int64_t stage_bytes = 0;
    int copies_per_thread = 0;
    int tile_height = 0;
    int64_t bytes_in_flight_per_sm = 0;
};

constexpr Specialization make_spec(int stages, int bif_kib) {
    Specialization s{};
    s.stages = stages;
    s.bif_kib = bif_kib;
    s.stage_bytes = (static_cast<int64_t>(bif_kib) * 1024) / stages;
    s.copies_per_thread = static_cast<int>(s.stage_bytes / (kThreadsPerCta * kVectorBytes));
    s.tile_height = static_cast<int>(s.stage_bytes / kTileWidthBytes);
    s.bytes_in_flight_per_sm = s.stage_bytes * stages;
    return s;
}

constexpr Specialization kSpecializations[9] = {
    make_spec(2, 16), make_spec(2, 32), make_spec(2, 64),
    make_spec(4, 16), make_spec(4, 32), make_spec(4, 64),
    make_spec(8, 16), make_spec(8, 32), make_spec(8, 64),
};

const Specialization& find_spec(int stages, int bif_kib) {
    for (const auto& s : kSpecializations) {
        if (s.stages == stages && s.bif_kib == bif_kib) return s;
    }
    fail("internal error: no specialization table entry for stages=%d bytes_in_flight_kib=%d",
         stages, bif_kib);
    std::abort();  // unreachable; fail() does not return.
}

// ---------------------------------------------------------------------------
// Device code: deterministic, index-verifiable pattern; init kernel; the
// shared cp.async emission helper; validate and benchmark kernel templates.
// ---------------------------------------------------------------------------

// SplitMix64's finalizer uses all 64 input bits and is a permutation over
// uint64_t. The first 64 bits of each vector therefore do not repeat before
// the index itself wraps; the second independently salted mix fills the
// remaining 64 bits. Validation never needs host-side source data.
__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ uint4 expected_vector(int64_t global_vec_index) {
    const uint64_t index = static_cast<uint64_t>(global_vec_index);
    const uint64_t lo = mix64(index);
    const uint64_t hi = mix64(index ^ kPatternSalt);
    return make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32),
                      static_cast<uint32_t>(hi), static_cast<uint32_t>(hi >> 32));
}

__global__ void init_pattern_kernel(uint4* __restrict__ g_src, int64_t total_vectors) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total_vectors; i += stride) {
        g_src[i] = expected_vector(i);
    }
}

// Issues COPIES 16-byte cp.async.cg.shared.global copies for one thread, one
// pipeline stage. Shared by the validate and benchmark kernels below so both
// execution paths emit their copies through the same helper.
template <int COPIES>
__device__ __forceinline__ void emit_stage_cp_async(
        const uint4* __restrict__ tile_src, unsigned char* __restrict__ slot, int tid) {
#pragma unroll
    for (int c = 0; c < COPIES; ++c) {
        const int vec = tid + c * kThreadsPerCta;
        const uint4* src_ptr = tile_src + vec;
        unsigned char* dst_ptr = slot + static_cast<size_t>(vec) * kVectorBytes;
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr), "l"(src_ptr), "n"(kVectorBytes)
                     : "memory");
    }
}

// Correctness path: walks every tile once (no rotation needed — rotation
// only permutes visit order within a pass, not the set of tiles visited),
// verifies every copied 16-byte vector, and accumulates mismatches locally
// before a single atomicAdd, so the working set never has to travel back to
// the host.
template <int STAGES, int COPIES>
__global__ void ldgsts_validate_kernel(
        const uint4* __restrict__ g_src,
        int64_t tiles_per_cta,
        unsigned long long* __restrict__ g_mismatch_count) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kElemsPerTile = static_cast<int64_t>(kThreadsPerCta) * COPIES;
    constexpr int64_t kStageBytes = kElemsPerTile * kVectorBytes;

    const int tid = threadIdx.x;
    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    unsigned long long local_mismatches = 0;

    auto consume = [&](int64_t consume_s) {
        const int64_t tile_idx = cta_tile_base + consume_s;
        const int64_t tile_vec_base = tile_idx * kElemsPerTile;
        unsigned char* slot = smem + (consume_s % STAGES) * kStageBytes;
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
            const int vec = tid + c * kThreadsPerCta;
            const uint4 got = *reinterpret_cast<uint4*>(slot + static_cast<size_t>(vec) * kVectorBytes);
            const uint4 want = expected_vector(tile_vec_base + vec);
            if (got.x != want.x || got.y != want.y || got.z != want.z || got.w != want.w) {
                ++local_mismatches;
            }
        }
    };

    for (int64_t s = 0; s < tiles_per_cta; ++s) {
        const uint4* tile_src = g_src + (cta_tile_base + s) * kElemsPerTile;
        unsigned char* slot = smem + (s % STAGES) * kStageBytes;
        emit_stage_cp_async<COPIES>(tile_src, slot, tid);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (s >= STAGES - 1) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1) : "memory");
            __syncthreads();
            consume(s - (STAGES - 1));
            __syncthreads();
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    int64_t drain_start = tiles_per_cta - STAGES + 1;
    if (drain_start < 0) drain_start = 0;
    for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
        consume(consume_s);
    }
    __syncthreads();

    if (local_mismatches != 0) {
        atomicAdd(g_mismatch_count, local_mismatches);
    }
}

// Timed path: identical pipeline mechanics, but instead of comparing against
// the expected pattern it XORs one 32-bit lane per thread into a tiny
// per-thread sink buffer. The sink write is a minimal observable global
// store that keeps the compiler from discarding the pipeline; its own
// traffic (grid_blocks*128*4 bytes, independent of the working set) is never
// counted toward useful_bytes/effective_gbps.
template <int STAGES, int COPIES>
__global__ void ldgsts_benchmark_kernel(
        const uint4* __restrict__ g_src,
        uint32_t* __restrict__ g_sink,
        int64_t tiles_per_cta,
        int64_t passes,
        int64_t rotation_base) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kElemsPerTile = static_cast<int64_t>(kThreadsPerCta) * COPIES;
    constexpr int64_t kStageBytes = kElemsPerTile * kVectorBytes;

    const int tid = threadIdx.x;
    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    uint32_t sink_acc = 0;

    for (int64_t p = 0; p < passes; ++p) {
        const int64_t rotation = (rotation_base + p) % tiles_per_cta;

        auto touch = [&](int64_t consume_s) {
            unsigned char* slot = smem + (consume_s % STAGES) * kStageBytes;
            sink_acc ^= *reinterpret_cast<uint32_t*>(slot + static_cast<size_t>(tid) * kVectorBytes);
        };

        for (int64_t s = 0; s < tiles_per_cta; ++s) {
            const int64_t tile_idx = cta_tile_base + ((rotation + s) % tiles_per_cta);
            const uint4* tile_src = g_src + tile_idx * kElemsPerTile;
            unsigned char* slot = smem + (s % STAGES) * kStageBytes;
            emit_stage_cp_async<COPIES>(tile_src, slot, tid);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            if (s >= STAGES - 1) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1) : "memory");
                __syncthreads();
                touch(s - (STAGES - 1));
                __syncthreads();
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        int64_t drain_start = tiles_per_cta - STAGES + 1;
        if (drain_start < 0) drain_start = 0;
        for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
            touch(consume_s);
        }
        __syncthreads();
    }

    g_sink[static_cast<int64_t>(blockIdx.x) * kThreadsPerCta + tid] = sink_acc;
}

// ---------------------------------------------------------------------------
// Host: device query, working-set planning, CSV, CLI, orchestration.
// ---------------------------------------------------------------------------
struct GpuInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    int64_t l2_bytes = 0;
    int64_t smem_per_sm_bytes = 0;
    int64_t smem_optin_max_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

GpuInfo query_gpu_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fail("cudaGetDeviceCount failed: %s (%s)", cudaGetErrorName(err), cudaGetErrorString(err));
    }
    if (device_count != 1) {
        fail("expected exactly 1 visible CUDA device, found %d", device_count);
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fail("cudaSetDevice(0) failed: %s (%s)", cudaGetErrorName(err), cudaGetErrorString(err));
    }
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fail("cudaGetDeviceProperties failed: %s (%s)", cudaGetErrorName(err), cudaGetErrorString(err));
    }
    if (prop.major != 10 || prop.minor != 3) {
        fail("expected compute capability 10.3, found %d.%d", prop.major, prop.minor);
    }
    int optin = 0;
    err = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (err != cudaSuccess) {
        fail("cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin) failed: %s (%s)",
             cudaGetErrorName(err), cudaGetErrorString(err));
    }
    int driver_version = 0, runtime_version = 0;
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) fail("cudaDriverGetVersion failed: %s", cudaGetErrorName(err));
    err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) fail("cudaRuntimeGetVersion failed: %s", cudaGetErrorName(err));

    if (prop.multiProcessorCount <= 0) fail("invalid multiProcessorCount=%d", prop.multiProcessorCount);
    if (prop.l2CacheSize <= 0) fail("invalid l2CacheSize=%d", prop.l2CacheSize);
    if (prop.sharedMemPerMultiprocessor <= 0) fail("invalid sharedMemPerMultiprocessor");

    GpuInfo info;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.l2_bytes = static_cast<int64_t>(prop.l2CacheSize);
    info.smem_per_sm_bytes = static_cast<int64_t>(prop.sharedMemPerMultiprocessor);
    info.smem_optin_max_bytes = static_cast<int64_t>(optin);
    info.driver_version = driver_version;
    info.runtime_version = runtime_version;
    return info;
}

int64_t round_up_to_multiple(int64_t value, int64_t multiple) {
    if (value <= 0) return multiple;
    const int64_t units = (value + multiple - 1) / multiple;
    return units * multiple;
}

struct WorkingSetPlan {
    int64_t requested_bytes = 0;
    int64_t working_set_bytes = 0;
    int64_t common_multiple_bytes = 0;
};

WorkingSetPlan plan_working_set(const GpuInfo& gpu, std::optional<int64_t> requested_mib) {
    const int64_t common_multiple = static_cast<int64_t>(gpu.sm_count) * 32 * 1024;
    int64_t requested_bytes;
    if (requested_mib.has_value()) {
        requested_bytes = requested_mib.value() * int64_t(1024) * 1024;
    } else {
        requested_bytes = int64_t(4) * gpu.l2_bytes;  // default: at least 4x L2
    }
    WorkingSetPlan plan;
    plan.requested_bytes = requested_bytes;
    plan.working_set_bytes = round_up_to_multiple(requested_bytes, common_multiple);
    plan.common_multiple_bytes = common_multiple;
    return plan;
}

WorkingSetPlan plan_self_test_working_set(const GpuInfo& gpu) {
    const int64_t common_multiple = static_cast<int64_t>(gpu.sm_count) * 32 * 1024;
    const int64_t working_set_bytes = common_multiple * kSelfTestCommonMultiples;
    return {working_set_bytes, working_set_bytes, common_multiple};
}

std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

std::string now_utc_iso8601() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

std::string run_command_capture(const char* cmd) {
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    const int rc = pclose(pipe);
    if (rc != 0) return "";
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
}

std::string git_commit_hash() {
    const std::string out = run_command_capture("git rev-parse HEAD 2>/dev/null");
    return out.empty() ? "UNKNOWN" : out;
}

std::string git_dirty_flag() {
    FILE* pipe = popen("git status --porcelain 2>/dev/null", "r");
    if (!pipe) return "unknown";
    char buffer[256];
    bool any = false;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        any = true;
    }
    const int rc = pclose(pipe);
    if (rc != 0) return "unknown";
    return any ? "true" : "false";
}

struct CliConfig {
    bool help = false;
    bool self_test = false;
    bool has_stages = false;
    int stages = 0;
    bool has_bif = false;
    int bif_kib = 0;
    bool has_working_set_mib = false;
    int64_t working_set_mib = 0;
    bool has_passes = false;
    int64_t passes = 1;
    bool has_warmup_ms = false;
    int64_t warmup_ms = 0;
    bool has_repetitions = false;
    int64_t repetitions = 1;
    bool has_run_kind = false;
    std::string run_kind;
};

void print_usage(std::FILE* out) {
    std::fprintf(out,
        "ldgsts - standalone LDGSTS (cp.async) global-memory->SMEM copy microbenchmark\n"
        "\n"
        "LDGSTS arm of the LDGSTS vs TMA experiment. Measures the effective copy\n"
        "bandwidth of a vectorized cp.async.cg.shared.global software pipeline\n"
        "(16-byte copies, one ring of pipeline stages). Not TMA, not DRAM/HBM\n"
        "bandwidth, not a final benchmark result.\n"
        "\n"
        "Usage:\n"
        "  ldgsts --self-test\n"
        "  ldgsts --stages {2,4,8} --bytes-in-flight-kib {16,32,64} --run-kind {smoke,benchmark}\n"
        "         [--working-set-mib N] [--passes N] [--warmup-ms N] [--repetitions N]\n"
        "\n"
        "Options:\n"
        "  --stages {2,4,8}                Ring buffer depth (pipeline stages). Required.\n"
        "  --bytes-in-flight-kib {16,32,64} Bytes in flight per SM. Required.\n"
        "  --working-set-mib N              Requested working set in MiB, in [1, 1048576].\n"
        "                                   Rounded up to a common multiple of\n"
        "                                   sm_count*32KiB. Default: at least 4x the\n"
        "                                   queried L2 cache size.\n"
        "  --passes N                       Full working-set traversals per measured\n"
        "                                   kernel launch, in [1, 1000000]. Default: 1.\n"
        "  --warmup-ms N                    Minimum warm-up time in ms before timed\n"
        "                                   repetitions begin, in [0, 3600000]. Default: 0.\n"
        "  --repetitions N                  Separately timed kernel launches, in\n"
        "                                   [1, 1000000]. Default: 1.\n"
        "  --run-kind {smoke,benchmark}     'benchmark' requires working_set_bytes > 2xL2.\n"
        "                                   'smoke' has no such requirement and its output\n"
        "                                   is never a final experimental result. Required.\n"
        "  --self-test                      Validate all nine specializations on a small\n"
        "                                   fixed working set and exit; no CSV, no timing.\n"
        "                                   Cannot be combined with the flags above.\n"
        "  --help                           Show this help and exit.\n"
        "\n"
        "On a --stages/--bytes-in-flight-kib/--run-kind run, stdout carries only CSV (one\n"
        "header line plus one row per repetition); diagnostics, progress, and errors go to\n"
        "stderr. See README.md for the CSV schema and units.\n");
}

bool parse_int_arg(const std::string& s, int64_t* out) {
    if (s.empty()) return false;
    errno = 0;
    char* endptr = nullptr;
    const long long v = std::strtoll(s.c_str(), &endptr, 10);
    if (errno != 0 || endptr == s.c_str() || *endptr != '\0') return false;
    *out = static_cast<int64_t>(v);
    return true;
}

bool parse_cli(int argc, char** argv, CliConfig* cfg, std::string* err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](void) -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h") {
            cfg->help = true;
            continue;
        }
        if (arg == "--self-test") {
            cfg->self_test = true;
            continue;
        }
        if (arg == "--stages") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || (iv != 2 && iv != 4 && iv != 8)) {
                *err = "--stages must be one of 2, 4, 8";
                return false;
            }
            cfg->stages = static_cast<int>(iv);
            cfg->has_stages = true;
            continue;
        }
        if (arg == "--bytes-in-flight-kib") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || (iv != 16 && iv != 32 && iv != 64)) {
                *err = "--bytes-in-flight-kib must be one of 16, 32, 64";
                return false;
            }
            cfg->bif_kib = static_cast<int>(iv);
            cfg->has_bif = true;
            continue;
        }
        if (arg == "--working-set-mib") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 1 || iv > (int64_t(1) << 20)) {
                *err = "--working-set-mib must be an integer in [1, 1048576]";
                return false;
            }
            cfg->working_set_mib = iv;
            cfg->has_working_set_mib = true;
            continue;
        }
        if (arg == "--passes") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 1 || iv > 1000000) {
                *err = "--passes must be an integer in [1, 1000000]";
                return false;
            }
            cfg->passes = iv;
            cfg->has_passes = true;
            continue;
        }
        if (arg == "--warmup-ms") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 0 || iv > 3600000) {
                *err = "--warmup-ms must be an integer in [0, 3600000]";
                return false;
            }
            cfg->warmup_ms = iv;
            cfg->has_warmup_ms = true;
            continue;
        }
        if (arg == "--repetitions") {
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 1 || iv > 1000000) {
                *err = "--repetitions must be an integer in [1, 1000000]";
                return false;
            }
            cfg->repetitions = iv;
            cfg->has_repetitions = true;
            continue;
        }
        if (arg == "--run-kind") {
            const auto v = next_value();
            if (!v || (*v != "smoke" && *v != "benchmark")) {
                *err = "--run-kind must be 'smoke' or 'benchmark'";
                return false;
            }
            cfg->run_kind = *v;
            cfg->has_run_kind = true;
            continue;
        }
        *err = "unknown argument: " + arg;
        return false;
    }

    if (cfg->help) return true;

    if (cfg->self_test) {
        if (cfg->has_stages || cfg->has_bif || cfg->has_working_set_mib || cfg->has_passes ||
            cfg->has_warmup_ms || cfg->has_repetitions || cfg->has_run_kind) {
            *err = "--self-test cannot be combined with benchmark options";
            return false;
        }
        return true;
    }

    if (!cfg->has_stages) { *err = "--stages is required (unless --self-test)"; return false; }
    if (!cfg->has_bif) { *err = "--bytes-in-flight-kib is required (unless --self-test)"; return false; }
    if (!cfg->has_run_kind) { *err = "--run-kind is required (unless --self-test)"; return false; }
    return true;
}

struct CsvRow {
    std::string timestamp_utc;
    std::string run_kind;
    int64_t sample_index = 0;
    Specialization spec;
    int occupancy_ctas_per_sm = 0;
    int grid_blocks = 0;
    int sm_count = 0;
    int64_t smem_reservation_bytes = 0;
    int64_t l2_bytes = 0;
    int64_t requested_working_set_bytes = 0;
    int64_t working_set_bytes = 0;
    int64_t passes = 0;
    int64_t useful_bytes = 0;
    int64_t warmup_ms = 0;
    double kernel_time_ms = 0.0;
    double effective_gbps = 0.0;
    std::string correctness;
    unsigned long long mismatches = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::string compute_capability;
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
    std::string git_commit;
    std::string git_dirty;
};

void print_csv_header() {
    std::printf(
        "schema_version,timestamp_utc,run_kind,method,sample_index,stages,"
        "tile_width_elements,tile_width_bytes,tile_height,stage_bytes,"
        "bytes_in_flight_per_sm,vector_bytes,copies_per_thread_per_stage,"
        "threads_per_cta,target_ctas_per_sm,occupancy_ctas_per_sm,grid_blocks,"
        "sm_count,smem_reservation_bytes,l2_bytes,requested_working_set_bytes,"
        "working_set_bytes,working_set_l2_ratio,passes,useful_bytes,warmup_ms,"
        "kernel_time_ms,effective_gbps,correctness,mismatches,gpu_name,gpu_uuid,"
        "compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,"
        "git_dirty\n");
}

void print_csv_row(const CsvRow& r) {
    const double working_set_l2_ratio =
        r.l2_bytes > 0 ? static_cast<double>(r.working_set_bytes) / static_cast<double>(r.l2_bytes)
                       : 0.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << kSchemaVersion << ',' << r.timestamp_utc << ',' << r.run_kind << ',' << kMethodName
        << ',' << r.sample_index << ',' << r.spec.stages << ',' << kTileWidthElements << ','
        << kTileWidthBytes << ',' << r.spec.tile_height << ',' << r.spec.stage_bytes << ','
        << r.spec.bytes_in_flight_per_sm << ',' << kVectorBytes << ',' << r.spec.copies_per_thread
        << ',' << kThreadsPerCta << ',' << kTargetMaxActiveCtasPerSm << ','
        << r.occupancy_ctas_per_sm
        << ',' << r.grid_blocks << ',' << r.sm_count << ',' << r.smem_reservation_bytes << ','
        << r.l2_bytes << ',' << r.requested_working_set_bytes << ',' << r.working_set_bytes << ','
        << working_set_l2_ratio << ',' << r.passes << ',' << r.useful_bytes << ',' << r.warmup_ms
        << ',' << r.kernel_time_ms << ',' << r.effective_gbps << ',' << r.correctness << ','
        << r.mismatches << ',' << csv_quote(r.gpu_name) << ',' << r.gpu_uuid << ','
        << r.compute_capability << ',' << r.cuda_driver_version << ',' << r.cuda_runtime_version
        << ',' << r.git_commit << ',' << r.git_dirty << '\n';
    std::fputs(oss.str().c_str(), stdout);
}

// Runs one (STAGES, COPIES) specialization: computes and verifies the
// max-active-CTAs/SM=1 shared-memory reservation, allocates and initializes
// the working set, validates correctness, and — only if validation passed
// and the caller asked for it — runs warm-up plus timed repetitions, printing
// one CSV row per repetition. Mismatches and CUDA failures remain distinct.
template <int STAGES, int COPIES>
RunStatus run_specialization(
        const GpuInfo& gpu,
        const Specialization& spec,
        const WorkingSetPlan& ws,
        const CliConfig& cli,
        bool benchmark_after_validate,
        bool print_header,
        const std::string& git_commit,
        const std::string& git_dirty,
        const std::string& gpu_uuid,
        uint64_t* out_mismatches) {
    static_assert(STAGES == 2 || STAGES == 4 || STAGES == 8, "invalid STAGES");
    if (out_mismatches) *out_mismatches = 0;

    const int grid_blocks = gpu.sm_count;
    const int64_t per_cta_bytes = ws.working_set_bytes / grid_blocks;
    const int64_t tiles_per_cta = per_cta_bytes / spec.stage_bytes;
    if (tiles_per_cta < 1) {
        fail("stages=%d bytes_in_flight_kib=%d: working_set_bytes=%lld yields 0 tiles/CTA "
             "(per_cta_bytes=%lld, stage_bytes=%lld)",
             spec.stages, spec.bif_kib, (long long)ws.working_set_bytes,
             (long long)per_cta_bytes, (long long)spec.stage_bytes);
    }

    // Shared-memory reservation: strictly more than half of
    // sharedMemPerMultiprocessor (so the resource limit permits at most one
    // active CTA per SM) and at least bytes_in_flight_per_sm, aligned to
    // kSmemAlignmentBytes and capped at the max opt-in value. This is a
    // residency limit, not an observation of runtime block placement.
    int64_t half_plus = (gpu.smem_per_sm_bytes / 2) + 1;
    half_plus = round_up_to_multiple(half_plus, kSmemAlignmentBytes);
    const int64_t bif_aligned = round_up_to_multiple(spec.bytes_in_flight_per_sm, kSmemAlignmentBytes);
    const int64_t reservation = std::max(half_plus, bif_aligned);
    if (reservation > gpu.smem_optin_max_bytes) {
        fail("stages=%d bytes_in_flight_kib=%d: required smem reservation %lld exceeds max "
             "opt-in %lld",
             spec.stages, spec.bif_kib, (long long)reservation, (long long)gpu.smem_optin_max_bytes);
    }

    CUDA_CHECK(cudaFuncSetAttribute(ldgsts_validate_kernel<STAGES, COPIES>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(reservation)));
    CUDA_CHECK(cudaFuncSetAttribute(ldgsts_benchmark_kernel<STAGES, COPIES>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(reservation)));

    int occ_validate = 0, occ_benchmark = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_validate, ldgsts_validate_kernel<STAGES, COPIES>, kThreadsPerCta,
        static_cast<size_t>(reservation)));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_benchmark, ldgsts_benchmark_kernel<STAGES, COPIES>, kThreadsPerCta,
        static_cast<size_t>(reservation)));
    if (occ_validate != kTargetMaxActiveCtasPerSm ||
        occ_benchmark != kTargetMaxActiveCtasPerSm) {
        fail("stages=%d bytes_in_flight_kib=%d: occupancy check failed (validate=%d "
             "benchmark=%d, need max_active_ctas_per_sm=%d); smem_reservation_bytes=%lld "
             "smem_per_sm_bytes=%lld",
             spec.stages, spec.bif_kib, occ_validate, occ_benchmark,
             kTargetMaxActiveCtasPerSm,
             (long long)reservation, (long long)gpu.smem_per_sm_bytes);
    }

    const int64_t total_vectors = ws.working_set_bytes / kVectorBytes;

    DeviceBuffer<uint4> d_src("src");
    CUDA_CHECK(d_src.allocate(static_cast<size_t>(total_vectors)));

    {
        constexpr int kInitThreads = 256;
        int64_t desired_blocks = (total_vectors + kInitThreads - 1) / kInitThreads;
        int64_t capped_blocks = std::min<int64_t>(desired_blocks, static_cast<int64_t>(grid_blocks) * 8);
        if (capped_blocks < 1) capped_blocks = 1;
        init_pattern_kernel<<<static_cast<unsigned int>(capped_blocks), kInitThreads>>>(
            d_src.get(), total_vectors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    DeviceBuffer<unsigned long long> d_mismatch("mismatch_count");
    CUDA_CHECK(d_mismatch.allocate(1));
    CUDA_CHECK(cudaMemset(d_mismatch.get(), 0, sizeof(unsigned long long)));

    ldgsts_validate_kernel<STAGES, COPIES>
        <<<grid_blocks, kThreadsPerCta, static_cast<size_t>(reservation)>>>(
            d_src.get(), tiles_per_cta, d_mismatch.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_mismatch = 0;
    CUDA_CHECK(cudaMemcpy(&h_mismatch, d_mismatch.get(), sizeof(h_mismatch), cudaMemcpyDeviceToHost));
    if (out_mismatches) *out_mismatches = h_mismatch;
    const bool validate_ok = (h_mismatch == 0);

    if (!validate_ok) return RunStatus::kMismatch;
    if (!benchmark_after_validate) return RunStatus::kOk;

    DeviceBuffer<uint32_t> d_sink("sink");
    CUDA_CHECK(d_sink.allocate(static_cast<size_t>(grid_blocks) * kThreadsPerCta));

    CudaEvent ev_start, ev_stop;
    CUDA_CHECK(ev_start.create());
    CUDA_CHECK(ev_stop.create());

    const auto warmup_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warmup_start)
               .count() < static_cast<double>(cli.warmup_ms)) {
        ldgsts_benchmark_kernel<STAGES, COPIES>
            <<<grid_blocks, kThreadsPerCta, static_cast<size_t>(reservation)>>>(
                d_src.get(), d_sink.get(), tiles_per_cta, cli.passes, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (print_header) print_csv_header();

    const int64_t useful_bytes = ws.working_set_bytes * cli.passes;

    for (int64_t rep = 0; rep < cli.repetitions; ++rep) {
        const int64_t rotation_base = rep % tiles_per_cta;
        CUDA_CHECK(cudaEventRecord(ev_start.get()));
        ldgsts_benchmark_kernel<STAGES, COPIES>
            <<<grid_blocks, kThreadsPerCta, static_cast<size_t>(reservation)>>>(
                d_src.get(), d_sink.get(), tiles_per_cta, cli.passes, rotation_base);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop.get()));
        CUDA_CHECK(cudaEventSynchronize(ev_stop.get()));

        float kernel_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, ev_start.get(), ev_stop.get()));

        const double kernel_time_seconds = static_cast<double>(kernel_time_ms) / 1000.0;
        const double effective_gbps = kernel_time_seconds > 0.0
            ? static_cast<double>(useful_bytes) / kernel_time_seconds / 1e9
            : 0.0;

        CsvRow row;
        row.timestamp_utc = now_utc_iso8601();
        row.run_kind = cli.run_kind;
        row.sample_index = rep;
        row.spec = spec;
        row.occupancy_ctas_per_sm = occ_benchmark;
        row.grid_blocks = grid_blocks;
        row.sm_count = gpu.sm_count;
        row.smem_reservation_bytes = reservation;
        row.l2_bytes = gpu.l2_bytes;
        row.requested_working_set_bytes = ws.requested_bytes;
        row.working_set_bytes = ws.working_set_bytes;
        row.passes = cli.passes;
        row.useful_bytes = useful_bytes;
        row.warmup_ms = cli.warmup_ms;
        row.kernel_time_ms = static_cast<double>(kernel_time_ms);
        row.effective_gbps = effective_gbps;
        row.correctness = "OK";
        row.mismatches = h_mismatch;
        row.gpu_name = gpu.name;
        row.gpu_uuid = gpu_uuid;
        row.compute_capability = std::to_string(gpu.major) + "." + std::to_string(gpu.minor);
        row.cuda_driver_version = gpu.driver_version;
        row.cuda_runtime_version = gpu.runtime_version;
        row.git_commit = git_commit;
        row.git_dirty = git_dirty;
        print_csv_row(row);
    }

    return RunStatus::kOk;
}

RunStatus dispatch_run(int stages, int bif_kib, const GpuInfo& gpu,
                       const Specialization& spec, const WorkingSetPlan& ws,
                       const CliConfig& cli, bool benchmark_after_validate,
                       bool print_header, const std::string& git_commit,
                       const std::string& git_dirty, const std::string& gpu_uuid,
                       uint64_t* out_mismatches) {
    if (stages == 2 && bif_kib == 16)
        return run_specialization<2, 4>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 2 && bif_kib == 32)
        return run_specialization<2, 8>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 2 && bif_kib == 64)
        return run_specialization<2, 16>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                          git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 4 && bif_kib == 16)
        return run_specialization<4, 2>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 4 && bif_kib == 32)
        return run_specialization<4, 4>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 4 && bif_kib == 64)
        return run_specialization<4, 8>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 8 && bif_kib == 16)
        return run_specialization<8, 1>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 8 && bif_kib == 32)
        return run_specialization<8, 2>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    if (stages == 8 && bif_kib == 64)
        return run_specialization<8, 4>(gpu, spec, ws, cli, benchmark_after_validate, print_header,
                                         git_commit, git_dirty, gpu_uuid, out_mismatches);
    fail("internal error: no specialization for stages=%d bytes_in_flight_kib=%d", stages, bif_kib);
    std::abort();  // unreachable; fail() does not return.
}

RunStatus run_self_test(const GpuInfo& gpu) {
    std::fprintf(stderr, "ldgsts: SELF_TEST start\n");
    const WorkingSetPlan ws = plan_self_test_working_set(gpu);
    std::fprintf(stderr, "ldgsts: SELF_TEST working_set_bytes=%lld sm_count=%d\n",
                 (long long)ws.working_set_bytes, gpu.sm_count);
    const CliConfig dummy;
    RunStatus overall_status = RunStatus::kOk;
    for (const auto& spec : kSpecializations) {
        uint64_t mismatches = 0;
        const RunStatus status = dispatch_run(
            spec.stages, spec.bif_kib, gpu, spec, ws, dummy,
            /*benchmark_after_validate=*/false, /*print_header=*/false,
            "", "", "", &mismatches);
        std::fprintf(stderr,
            "ldgsts: SELF_TEST stages=%d bytes_in_flight_kib=%d stage_bytes=%lld "
            "copies_per_thread_per_stage=%d result=%s mismatches=%llu\n",
            spec.stages, spec.bif_kib, (long long)spec.stage_bytes, spec.copies_per_thread,
            run_status_name(status), (unsigned long long)mismatches);
        if (status == RunStatus::kCudaError) {
            std::fprintf(stderr, "ldgsts: SELF_TEST_RESULT=CUDA_ERROR\n");
            return status;
        }
        if (status == RunStatus::kMismatch) overall_status = status;
    }
    std::fprintf(stderr, "ldgsts: SELF_TEST_RESULT=%s\n", run_status_name(overall_status));
    return overall_status;
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig cli;
    std::string parse_err;
    if (!parse_cli(argc, argv, &cli, &parse_err)) {
        std::fprintf(stderr, "ldgsts: ERROR: %s\n", parse_err.c_str());
        print_usage(stderr);
        return 2;
    }
    if (cli.help) {
        print_usage(stdout);
        return 0;
    }

    // No CUDA calls above this point, so --help and argument-validation
    // errors work in a GPU-less environment (e.g. inside the build
    // container during static/CLI checks).
    const GpuInfo gpu = query_gpu_info();

    const char* uuid_env = std::getenv("EXPECTED_GPU_UUID");
    if (uuid_env == nullptr || uuid_env[0] == '\0') {
        fail("EXPECTED_GPU_UUID is not set; run this binary only via scripts/run_gpu.sh");
    }
    const std::string gpu_uuid(uuid_env);

    int overall_rc = 0;

    if (cli.self_test) {
        overall_rc = run_self_test(gpu) == RunStatus::kOk ? 0 : 1;
    } else {
        const Specialization& spec = find_spec(cli.stages, cli.bif_kib);
        const WorkingSetPlan ws = plan_working_set(
            gpu, cli.has_working_set_mib ? std::optional<int64_t>(cli.working_set_mib) : std::nullopt);

        std::fprintf(stderr,
            "ldgsts: run_kind=%s stages=%d bytes_in_flight_kib=%d "
            "requested_working_set_bytes=%lld working_set_bytes=%lld l2_bytes=%lld "
            "common_multiple_bytes=%lld\n",
            cli.run_kind.c_str(), cli.stages, cli.bif_kib, (long long)ws.requested_bytes,
            (long long)ws.working_set_bytes, (long long)gpu.l2_bytes,
            (long long)ws.common_multiple_bytes);

        if (cli.run_kind == "benchmark" && ws.working_set_bytes <= 2 * gpu.l2_bytes) {
            fail("working_set_bytes=%lld is not strictly greater than 2xL2 (%lld); increase "
                 "--working-set-mib for a benchmark run",
                 (long long)ws.working_set_bytes, (long long)(2 * gpu.l2_bytes));
        }

        const std::string git_commit_str = git_commit_hash();
        const std::string git_dirty_str = git_dirty_flag();

        uint64_t mismatches = 0;
        const RunStatus status = dispatch_run(
            cli.stages, cli.bif_kib, gpu, spec, ws, cli,
            /*benchmark_after_validate=*/true, /*print_header=*/true,
            git_commit_str, git_dirty_str, gpu_uuid, &mismatches);
        if (status == RunStatus::kMismatch) {
            std::fprintf(stderr,
                "ldgsts: ERROR: correctness validation FAILED for stages=%d "
                "bytes_in_flight_kib=%d mismatches=%llu; no benchmark was run\n",
                cli.stages, cli.bif_kib, (unsigned long long)mismatches);
            overall_rc = 1;
        } else if (status == RunStatus::kCudaError) {
            std::fprintf(stderr,
                "ldgsts: ERROR: execution aborted by a CUDA error for stages=%d "
                "bytes_in_flight_kib=%d; discard any partial CSV output\n",
                cli.stages, cli.bif_kib);
            overall_rc = 1;
        } else {
            std::fprintf(stderr, "ldgsts: correctness=OK mismatches=0; benchmark complete\n");
        }
    }

    if (g_cleanup_failures != 0) {
        std::fprintf(stderr, "ldgsts: ERROR: %d resource cleanup failure(s) occurred\n",
                     g_cleanup_failures);
        overall_rc = 1;
    }
    return overall_rc;
}
