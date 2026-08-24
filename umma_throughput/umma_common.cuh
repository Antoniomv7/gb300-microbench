#pragma once

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda/ptx>

#include "../benchmark_common.cuh"

namespace {

constexpr int kThreadsPerCta = 128;
constexpr int kClusterCtas = 2;
constexpr int kK = 16;
constexpr const char* kMethodName = UMMA_METHOD;

enum class TimingMode : int32_t { kUntimed = 0, kTimed = 1 };

[[noreturn]] inline void fail(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::fprintf(stderr, "%s: ", kMethodName);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
    std::exit(1);
}

#define CUDA_CHECK_FATAL(call)                                               \
    do {                                                                     \
        const cudaError_t error = (call);                                    \
        if (error != cudaSuccess) fail("%s", cudaGetErrorString(error));   \
    } while (0)

__device__ __forceinline__ int smem_core_tile_index(int group, int position, int k) {
    return group * 128 + (k / 8) * 64 + position * 8 + k % 8;
}

__device__ __forceinline__ uint64_t make_smem_descriptor(const void* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    return static_cast<uint64_t>((address >> 4) & 0x3FFFu) |
           (static_cast<uint64_t>(128 >> 4) << 16) |
           (static_cast<uint64_t>(256 >> 4) << 32) | (UINT64_C(1) << 46);
}

template <int M, int N>
__device__ constexpr uint32_t make_umma_descriptor() {
    return (1u << 4) | (1u << 7) | (1u << 10) |
           ((static_cast<uint32_t>(N) >> 3) << 17) |
           ((static_cast<uint32_t>(M) >> 4) << 24);
}

__device__ __forceinline__ void issue_one_umma_1sm(uint32_t d, uint64_t a, uint64_t b,
                                                   uint32_t descriptor, int accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t"
        "}\n\t"
        : : "r"(d), "l"(a), "l"(b), "r"(descriptor), "r"(accumulate) : "memory");
}

__device__ __forceinline__ void issue_one_umma_2sm(uint32_t d, uint64_t a, uint64_t b,
                                                   uint32_t descriptor, int accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t"
        "}\n\t"
        : : "r"(d), "l"(a), "l"(b), "r"(descriptor), "r"(accumulate) : "memory");
}

__device__ __forceinline__ void commit_umma_1sm(uint32_t barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 : : "r"(barrier) : "memory");
}

__device__ __forceinline__ void commit_umma_2sm_multicast(uint32_t barrier, uint16_t mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        : : "r"(barrier), "h"(mask) : "memory");
}

__device__ __forceinline__ void tcgen05_alloc_1sm(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(address), "r"(columns) : "memory");
}

__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(address), "r"(columns) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_1sm(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 : : "r"(address), "r"(columns) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 : : "r"(address), "r"(columns) : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_1sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t address, uint32_t out[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]),
          "=r"(out[6]), "=r"(out[7]), "=r"(out[8]), "=r"(out[9]), "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15]), "=r"(out[16]), "=r"(out[17]),
          "=r"(out[18]), "=r"(out[19]), "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]),
          "=r"(out[24]), "=r"(out[25]), "=r"(out[26]), "=r"(out[27]), "=r"(out[28]), "=r"(out[29]),
          "=r"(out[30]), "=r"(out[31])
        : "r"(address));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ uint32_t make_tmem_load_address(uint32_t base, int warp, int fragment) {
    return base + ((static_cast<uint32_t>(warp) * 32) << 16) +
           static_cast<uint32_t>(fragment) * 32;
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

struct CliConfig {
    bool help = false;
    std::string run_kind;
    std::string campaign_kind = "none";
    int n = 0;
    int depth = 0;
    int64_t iterations = 0;
    int64_t warmup_iterations = 0;
    int64_t repetitions = 0;
};

inline void print_usage(std::FILE* output) {
    std::fprintf(output, "Usage: %s --run-kind KIND [--n N --depth N] --iterations N "
                         "--warmup-iterations N --repetitions N\n", kMethodName);
}

inline bool parse_cli(int argc, char** argv, CliConfig* config, std::string* error,
                      bool device_scaling = false) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") { config->help = true; continue; }
        if (++index >= argc) { *error = "missing value for " + option; return false; }
        const std::string value = argv[index];
        if (option == "--run-kind") { config->run_kind = value; continue; }
        if (option == "--campaign-kind") { config->campaign_kind = value; continue; }
        int64_t number = 0;
        if (!parse_int_arg(value, &number)) { *error = "invalid value for " + option; return false; }
        if (option == "--n") config->n = static_cast<int>(number);
        else if (option == "--depth") config->depth = static_cast<int>(number);
        else if (option == "--iterations") config->iterations = number;
        else if (option == "--warmup-iterations") config->warmup_iterations = number;
        else if (option == "--repetitions") config->repetitions = number;
        else { *error = "unknown argument: " + option; return false; }
    }
    if (config->help) return true;
    if ((config->run_kind != "smoke" && config->run_kind != "benchmark") ||
        config->iterations < 1 || config->warmup_iterations < 0 || config->repetitions < 1 ||
        (!device_scaling && (config->n != 64 && config->n != 128 && config->n != 256)) ||
        (!device_scaling && config->depth != 4 && config->depth != 16 &&
         config->depth != 64 && config->depth != 256)) {
        *error = "invalid UMMA configuration";
        return false;
    }
    return true;
}

using IsolatedKernel = void (*)(int64_t, TimingMode, float*, unsigned long long*);

struct IsolatedSpecialization {
    int n;
    int depth;
    IsolatedKernel kernel;
};

inline void validate_umma(const std::vector<float>& result, int m, int n, int depth) {
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            int expected = 0;
            for (int k = 0; k < kK; ++k)
                expected += (((row + 3 * k) % 7) - 3) * (((2 * k + column) % 5) - 2);
            if (result[static_cast<size_t>(row) * n + column] != expected * depth)
                fail("numerical validation failed at row=%d column=%d", row, column);
        }
    }
}

inline int run_isolated(const IsolatedSpecialization& spec, const CliConfig& config,
                        int m, int group, int smem_bytes) {
    const size_t elements = static_cast<size_t>(m) * spec.n;
    float* output = nullptr;
    unsigned long long* device_cycles = nullptr;
    CUDA_CHECK_FATAL(cudaMalloc(&output, elements * sizeof(float)));
    CUDA_CHECK_FATAL(cudaMalloc(&device_cycles, sizeof(unsigned long long)));

    const auto launch = [&](TimingMode mode, bool validate) {
        spec.kernel<<<group, kThreadsPerCta, smem_bytes>>>(config.iterations, mode,
                                                             output, device_cycles);
        CUDA_CHECK_FATAL(cudaGetLastError());
        CUDA_CHECK_FATAL(cudaDeviceSynchronize());
        if (validate) {
            std::vector<float> values(elements);
            CUDA_CHECK_FATAL(cudaMemcpy(values.data(), output, elements * sizeof(float),
                                         cudaMemcpyDeviceToHost));
            validate_umma(values, m, spec.n, spec.depth);
        }
    };

    launch(TimingMode::kUntimed, true);
    for (int64_t index = 0; index < config.warmup_iterations; ++index)
        launch(TimingMode::kUntimed, false);

    std::puts("method,sample_index,cta_group,m,n,depth,iterations,total_flops,"
              "elapsed_cycles,cycles_per_umma,flops_per_cycle,correctness");
    const int64_t total_umma = config.iterations * spec.depth;
    const int64_t total_flops = 2 * static_cast<int64_t>(m) * spec.n * kK * total_umma;
    for (int64_t sample = 0; sample < config.repetitions; ++sample) {
        launch(TimingMode::kTimed, false);
        unsigned long long cycles = 0;
        CUDA_CHECK_FATAL(cudaMemcpy(&cycles, device_cycles, sizeof(cycles), cudaMemcpyDeviceToHost));
        if (!cycles) fail("elapsed cycles must be positive");
        std::printf("%s,%lld,%d,%d,%d,%d,%lld,%lld,%llu,%.6f,%.6f,OK\n", kMethodName,
                    static_cast<long long>(sample), group, m, spec.n, spec.depth,
                    static_cast<long long>(config.iterations), static_cast<long long>(total_flops),
                    cycles, static_cast<double>(cycles) / total_umma,
                    static_cast<double>(total_flops) / cycles);
    }
    CUDA_CHECK_FATAL(cudaFree(output));
    CUDA_CHECK_FATAL(cudaFree(device_cycles));
    return 0;
}

}  // namespace
