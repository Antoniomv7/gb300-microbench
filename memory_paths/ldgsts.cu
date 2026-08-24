// LDGSTS global-to-shared copy; validate before CUDA-event timing.

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

#include "../benchmark_common.cuh"

namespace {

constexpr int kThreadsPerCta = 128;
constexpr int kVectorBytes = 16;
constexpr int kTargetMaxActiveCtasPerSm = 1;
constexpr int kTileWidthElements = 128;  // BF16-width (2-byte) elements/row
constexpr int kTileWidthBytes = 256;     // 128 * 2 bytes
constexpr int64_t kSmemAlignmentBytes = 128;
constexpr uint64_t kPatternSalt = 0xD1B54A32D192ED03ULL;
constexpr const char* kSchemaVersion = "1";
constexpr const char* kMethodName = "ldgsts";

int g_cleanup_failures = 0;

enum class RunStatus {
    kOk,
    kMismatch,
    kCudaError,
};

[[noreturn]] void fail(const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "ldgsts: ERROR: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

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

// Validate every copied vector against its deterministic source pattern.
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

// Match the TMA arm: one CTA per SM and the same in-flight byte grid.
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

struct GpuInfo {
    int sm_count = 0;
    int64_t l2_bytes = 0;
    int64_t smem_per_sm_bytes = 0;
    int64_t smem_optin_max_bytes = 0;
};

GpuInfo query_gpu_info() {
    const cudaDeviceProp prop = benchmark_device_properties();
    int optin = 0;
    const cudaError_t err = cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (err != cudaSuccess) {
        fail("cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin) failed: %s (%s)",
             cudaGetErrorName(err), cudaGetErrorString(err));
    }

    if (prop.multiProcessorCount <= 0) fail("invalid multiProcessorCount=%d", prop.multiProcessorCount);
    if (prop.l2CacheSize <= 0) fail("invalid l2CacheSize=%d", prop.l2CacheSize);
    if (prop.sharedMemPerMultiprocessor <= 0) fail("invalid sharedMemPerMultiprocessor");

    GpuInfo info;
    info.sm_count = prop.multiProcessorCount;
    info.l2_bytes = static_cast<int64_t>(prop.l2CacheSize);
    info.smem_per_sm_bytes = static_cast<int64_t>(prop.sharedMemPerMultiprocessor);
    info.smem_optin_max_bytes = static_cast<int64_t>(optin);
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

struct CliConfig {
    bool help = false;
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
        "Usage: ldgsts --stages {2,4,8} --bytes-in-flight-kib {16,32,64}\n"
        "       --run-kind {smoke,benchmark} [--working-set-mib N] [--passes N]\n"
        "       [--warmup-ms N] [--repetitions N]\n");
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

    if (!cfg->has_stages) { *err = "--stages is required"; return false; }
    if (!cfg->has_bif) { *err = "--bytes-in-flight-kib is required"; return false; }
    if (!cfg->has_run_kind) { *err = "--run-kind is required"; return false; }
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
};

void print_csv_header() {
    std::printf(
        "schema_version,timestamp_utc,run_kind,method,sample_index,stages,"
        "tile_width_elements,tile_width_bytes,tile_height,stage_bytes,"
        "bytes_in_flight_per_sm,vector_bytes,copies_per_thread_per_stage,"
        "threads_per_cta,target_ctas_per_sm,occupancy_ctas_per_sm,grid_blocks,"
        "sm_count,smem_reservation_bytes,l2_bytes,requested_working_set_bytes,"
        "working_set_bytes,working_set_l2_ratio,passes,useful_bytes,warmup_ms,"
        "kernel_time_ms,effective_gbps,correctness,mismatches\n");
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
        << r.mismatches << '\n';
    std::fputs(oss.str().c_str(), stdout);
}

template <int STAGES, int COPIES>
RunStatus run_specialization(
        const GpuInfo& gpu,
        const Specialization& spec,
        const WorkingSetPlan& ws,
        const CliConfig& cli,
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

    print_csv_header();

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
        print_csv_row(row);
    }

    return RunStatus::kOk;
}

RunStatus dispatch_run(int stages, int bif_kib, const GpuInfo& gpu,
                       const Specialization& spec, const WorkingSetPlan& ws,
                       const CliConfig& cli, uint64_t* out_mismatches) {
    if (stages == 2 && bif_kib == 16)
        return run_specialization<2, 4>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 2 && bif_kib == 32)
        return run_specialization<2, 8>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 2 && bif_kib == 64)
        return run_specialization<2, 16>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 4 && bif_kib == 16)
        return run_specialization<4, 2>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 4 && bif_kib == 32)
        return run_specialization<4, 4>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 4 && bif_kib == 64)
        return run_specialization<4, 8>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 8 && bif_kib == 16)
        return run_specialization<8, 1>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 8 && bif_kib == 32)
        return run_specialization<8, 2>(gpu, spec, ws, cli, out_mismatches);
    if (stages == 8 && bif_kib == 64)
        return run_specialization<8, 4>(gpu, spec, ws, cli, out_mismatches);
    fail("internal error: no specialization for stages=%d bytes_in_flight_kib=%d", stages, bif_kib);
    std::abort();  // unreachable; fail() does not return.
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

    const GpuInfo gpu = query_gpu_info();
    const Specialization& spec = find_spec(cli.stages, cli.bif_kib);
    const WorkingSetPlan ws = plan_working_set(
        gpu, cli.has_working_set_mib ? std::optional<int64_t>(cli.working_set_mib) : std::nullopt);
    if (cli.run_kind == "benchmark" && ws.working_set_bytes <= 2 * gpu.l2_bytes)
        fail("working set must exceed twice the L2 cache size");

    uint64_t mismatches = 0;
    const RunStatus status = dispatch_run(cli.stages, cli.bif_kib, gpu, spec, ws, cli, &mismatches);
    if (status != RunStatus::kOk || g_cleanup_failures != 0) {
        std::fprintf(stderr, "ldgsts: measurement failed; mismatches=%llu cleanup_errors=%d\n",
                     static_cast<unsigned long long>(mismatches), g_cleanup_failures);
        return 1;
    }
    return 0;
}
