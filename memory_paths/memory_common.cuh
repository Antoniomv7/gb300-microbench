#pragma once

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../benchmark_common.cuh"

namespace {

constexpr int kThreadsPerCta = 128;
constexpr int kVectorBytes = 16;
constexpr int kTileWidthElements = 128;
constexpr int kTileWidthBytes = 256;
constexpr int64_t kSmemAlignmentBytes = 128;
constexpr uint64_t kPatternSalt = 0xD1B54A32D192ED03ULL;
constexpr const char* kMethodName = MEMORY_METHOD;

enum class RunStatus { kOk, kMismatch, kCudaError };

[[noreturn]] inline void fail(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::fprintf(stderr, "%s: ", kMethodName);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    std::exit(1);
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t error = (call);                                    \
        if (error != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s: %s\n", kMethodName,                  \
                         cudaGetErrorString(error));                         \
            return RunStatus::kCudaError;                                     \
        }                                                                    \
    } while (0)

template <typename T>
class DeviceBuffer {
 public:
    ~DeviceBuffer() { if (pointer_) cudaFree(pointer_); }
    cudaError_t allocate(size_t count) { return cudaMalloc(&pointer_, count * sizeof(T)); }
    T* get() const { return pointer_; }

 private:
    T* pointer_ = nullptr;
};

class CudaEvent {
 public:
    ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
    cudaError_t create() { return cudaEventCreate(&event_); }
    cudaEvent_t get() const { return event_; }

 private:
    cudaEvent_t event_ = nullptr;
};

__host__ __device__ constexpr int64_t compute_stage_bytes(int copies) {
    return static_cast<int64_t>(kThreadsPerCta) * copies * kVectorBytes;
}

__host__ __device__ constexpr int64_t compute_tile_height(int64_t stage_bytes) {
    return stage_bytes / kTileWidthBytes;
}

struct Specialization {
    int stages;
    int bif_kib;
    int64_t stage_bytes;
    int copies_per_thread;
    int tile_height;
    int64_t bytes_in_flight_per_sm;
};

constexpr Specialization make_spec(int stages, int bif_kib) {
    const int64_t stage_bytes = static_cast<int64_t>(bif_kib) * 1024 / stages;
    return {stages, bif_kib, stage_bytes,
            static_cast<int>(stage_bytes / (kThreadsPerCta * kVectorBytes)),
            static_cast<int>(compute_tile_height(stage_bytes)), stage_bytes * stages};
}

constexpr Specialization kSpecializations[] = {
    make_spec(2, 16), make_spec(2, 32), make_spec(2, 64),
    make_spec(4, 16), make_spec(4, 32), make_spec(4, 64),
    make_spec(8, 16), make_spec(8, 32), make_spec(8, 64),
};

inline const Specialization& find_spec(int stages, int bif_kib) {
    for (const auto& spec : kSpecializations)
        if (spec.stages == stages && spec.bif_kib == bif_kib) return spec;
    fail("unsupported configuration: stages=%d, in-flight=%d KiB", stages, bif_kib);
}

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ uint4 expected_vector(int64_t index) {
    const uint64_t lo = mix64(static_cast<uint64_t>(index));
    const uint64_t hi = mix64(static_cast<uint64_t>(index) ^ kPatternSalt);
    return make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32),
                      static_cast<uint32_t>(hi), static_cast<uint32_t>(hi >> 32));
}

__global__ void init_pattern_kernel(uint4* source, int64_t count) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride)
        source[index] = expected_vector(index);
}

struct GpuInfo {
    int sm_count;
    int64_t l2_bytes;
    int64_t smem_per_sm_bytes;
    int64_t smem_optin_max_bytes;
};

inline GpuInfo query_gpu_info() {
    const cudaDeviceProp properties = benchmark_device_properties();
    int maximum = 0;
    if (cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
        != cudaSuccess)
        fail("cannot query shared-memory capacity");
    return {properties.multiProcessorCount, properties.l2CacheSize,
            static_cast<int64_t>(properties.sharedMemPerMultiprocessor), maximum};
}

inline int64_t round_up_to_multiple(int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

struct CliConfig {
    bool help = false;
    int stages = 0;
    int bif_kib = 0;
    int64_t working_set_mib = 0;
    int64_t passes = 1;
    int64_t warmup_ms = 0;
    int64_t repetitions = 1;
    std::string run_kind;
};

inline void print_usage(std::FILE* output) {
    std::fprintf(output, "Usage: %s --stages N --bytes-in-flight-kib N --run-kind KIND "
                         "[--working-set-mib N] [--passes N] [--warmup-ms N] "
                         "[--repetitions N]\n", kMethodName);
}

inline bool parse_cli(int argc, char** argv, CliConfig* config, std::string* error) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") { config->help = true; continue; }
        if (++index >= argc) { *error = "missing value for " + option; return false; }
        const std::string value = argv[index];
        if (option == "--run-kind") { config->run_kind = value; continue; }
        int64_t number = 0;
        if (!parse_int_arg(value, &number)) { *error = "invalid value for " + option; return false; }
        if (option == "--stages") config->stages = static_cast<int>(number);
        else if (option == "--bytes-in-flight-kib") config->bif_kib = static_cast<int>(number);
        else if (option == "--working-set-mib") config->working_set_mib = number;
        else if (option == "--passes") config->passes = number;
        else if (option == "--warmup-ms") config->warmup_ms = number;
        else if (option == "--repetitions") config->repetitions = number;
        else { *error = "unknown argument: " + option; return false; }
    }
    if (config->help) return true;
    if ((config->stages != 2 && config->stages != 4 && config->stages != 8) ||
        (config->bif_kib != 16 && config->bif_kib != 32 && config->bif_kib != 64) ||
        (config->run_kind != "smoke" && config->run_kind != "benchmark") ||
        config->passes < 1 || config->warmup_ms < 0 || config->repetitions < 1) {
        *error = "invalid benchmark configuration";
        return false;
    }
    return true;
}

struct WorkingSetPlan { int64_t working_set_bytes; };

inline WorkingSetPlan plan_working_set(const GpuInfo& gpu, int64_t requested_mib) {
    const int64_t requested = requested_mib ? requested_mib * 1024 * 1024 : 4 * gpu.l2_bytes;
    return {round_up_to_multiple(requested, static_cast<int64_t>(gpu.sm_count) * 32 * 1024)};
}

inline void print_csv_header() {
    std::puts("method,sample_index,stages,bytes_in_flight_per_sm,sm_count,"
              "working_set_bytes,passes,useful_bytes,kernel_time_ms,effective_gbps,correctness");
}

inline void print_csv_row(const Specialization& spec, const GpuInfo& gpu,
                          const WorkingSetPlan& working_set, const CliConfig& config,
                          int64_t sample, float kernel_ms) {
    const int64_t useful_bytes = working_set.working_set_bytes * config.passes;
    const double bandwidth = useful_bytes / static_cast<double>(kernel_ms) / 1e6;
    std::printf("%s,%lld,%d,%lld,%d,%lld,%lld,%lld,%.6f,%.6f,OK\n", kMethodName,
                static_cast<long long>(sample), spec.stages,
                static_cast<long long>(spec.bytes_in_flight_per_sm), gpu.sm_count,
                static_cast<long long>(working_set.working_set_bytes),
                static_cast<long long>(config.passes), static_cast<long long>(useful_bytes),
                static_cast<double>(kernel_ms), bandwidth);
}

template <typename ValidateKernel, typename BenchmarkKernel, typename Prepare,
          typename LaunchValidation, typename LaunchBenchmark>
RunStatus run_memory_case(const GpuInfo& gpu, const Specialization& spec,
                          const WorkingSetPlan& working_set, const CliConfig& config,
                          ValidateKernel validate_kernel, BenchmarkKernel benchmark_kernel,
                          int64_t barrier_bytes, Prepare prepare,
                          LaunchValidation launch_validation, LaunchBenchmark launch_benchmark) {
    const int blocks = gpu.sm_count;
    const int64_t tiles = working_set.working_set_bytes / blocks / spec.stage_bytes;
    const int64_t payload = round_up_to_multiple(spec.bytes_in_flight_per_sm + barrier_bytes,
                                                   kSmemAlignmentBytes);
    const int64_t half_sm = round_up_to_multiple(gpu.smem_per_sm_bytes / 2 + 1,
                                                  kSmemAlignmentBytes);
    const size_t reservation = static_cast<size_t>(std::max(payload, half_sm));
    if (tiles < 1 || static_cast<int64_t>(reservation) > gpu.smem_optin_max_bytes)
        fail("unsupported working set or shared-memory reservation");

    CUDA_CHECK(cudaFuncSetAttribute(validate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(reservation)));
    CUDA_CHECK(cudaFuncSetAttribute(benchmark_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(reservation)));
    int validate_occupancy = 0, benchmark_occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &validate_occupancy, validate_kernel, kThreadsPerCta, reservation));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &benchmark_occupancy, benchmark_kernel, kThreadsPerCta, reservation));
    if (validate_occupancy != 1 || benchmark_occupancy != 1)
        fail("both kernels must use one CTA per SM");

    const int64_t vectors = working_set.working_set_bytes / kVectorBytes;
    DeviceBuffer<uint4> source;
    DeviceBuffer<unsigned long long> mismatches;
    DeviceBuffer<uint32_t> sink;
    CUDA_CHECK(source.allocate(static_cast<size_t>(vectors)));
    const int init_blocks = static_cast<int>(std::min<int64_t>((vectors + 255) / 256, blocks * 8));
    init_pattern_kernel<<<init_blocks, 256>>>(source.get(), vectors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const RunStatus prepared = prepare(source.get());
    if (prepared != RunStatus::kOk) return prepared;

    CUDA_CHECK(mismatches.allocate(1));
    CUDA_CHECK(cudaMemset(mismatches.get(), 0, sizeof(unsigned long long)));
    launch_validation(source.get(), mismatches.get(), tiles, blocks, reservation);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long mismatch_count = 0;
    CUDA_CHECK(cudaMemcpy(&mismatch_count, mismatches.get(), sizeof(mismatch_count),
                          cudaMemcpyDeviceToHost));
    if (mismatch_count) return RunStatus::kMismatch;

    CUDA_CHECK(sink.allocate(static_cast<size_t>(blocks) * kThreadsPerCta));
    CudaEvent start, stop;
    CUDA_CHECK(start.create());
    CUDA_CHECK(stop.create());
    const auto warmup_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - warmup_start).count() < config.warmup_ms) {
        launch_benchmark(source.get(), sink.get(), tiles, config.passes, 0, blocks, reservation);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    print_csv_header();
    for (int64_t sample = 0; sample < config.repetitions; ++sample) {
        CUDA_CHECK(cudaEventRecord(start.get()));
        launch_benchmark(source.get(), sink.get(), tiles, config.passes,
                         sample % tiles, blocks, reservation);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop.get()));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float kernel_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start.get(), stop.get()));
        if (!(kernel_ms > 0)) fail("kernel duration must be positive");
        print_csv_row(spec, gpu, working_set, config, sample, kernel_ms);
    }
    return RunStatus::kOk;
}

}  // namespace
