// Compare isolated and whole-device UMMA with CUDA-event timing.

#define UMMA_METHOD "umma_device_scaling"
#include "umma_common.cuh"

namespace {

constexpr int kN = 256;
constexpr int kDepth = 256;
constexpr int kMLocal = 128;
constexpr int kM1sm = 128;
constexpr int kM2sm = 256;
constexpr unsigned long long kResidencyTimeoutCycles = 2000000000ULL;

enum class RunMode : int32_t { kCompute = 0, kResidency = 1 };

__device__ __forceinline__ int current_smid() {
    unsigned int smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return static_cast<int>(smid);
}

__device__ __forceinline__ void residency_handshake(unsigned int* arrivals,
                                                     int* resident, int* smids) {
    if (threadIdx.x != 0) return;
    // Every CTA must remain resident until the full launch has reported its SM.
    smids[blockIdx.x] = current_smid();
    atomicAdd(arrivals, 1u);
    const unsigned long long started = clock64();
    while (atomicAdd(arrivals, 0u) < gridDim.x) {
        if (static_cast<unsigned long long>(clock64() - started) > kResidencyTimeoutCycles) {
            resident[blockIdx.x] = 0;
            return;
        }
    }
    resident[blockIdx.x] = 1;
}

}  // namespace

extern "C" __global__ __launch_bounds__(128)
void umma_1sm_scaling_m128n256k16_d256(int64_t iterations, RunMode mode,
                                        float* __restrict__ output,
                                        unsigned int* __restrict__ arrivals,
                                        int* __restrict__ resident,
                                        int* __restrict__ smids) {
    const int tid = threadIdx.x;
    const int block = static_cast<int>(blockIdx.x);
    if (mode == RunMode::kResidency) {
        residency_handshake(arrivals, resident, smids);
        return;
    }

    constexpr int kABytes = kMLocal * kK * 2;
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);
    const int warp_id = tid / 32;

    for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) {
        const int row = idx / kK;
        const int k = idx % kK;
        const int value = ((row + 3 * k) % 7) - 3;
        A[smem_core_tile_index(row / 8, row % 8, k)] = __float2bfloat16(static_cast<float>(value));
    }
    for (int idx = tid; idx < kN * kK; idx += kThreadsPerCta) {
        const int column = idx / kK;
        const int k = idx % kK;
        const int value = ((2 * k + column) % 5) - 2;
        B[smem_core_tile_index(column / 8, column % 8, k)] =
            __float2bfloat16(static_cast<float>(value));
    }

    __shared__ uint64_t barrier;
    __shared__ int tmem_address;
    __syncthreads();
    if (tid == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1u);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    bool leader = false;
    if (tid < 32) leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    if (warp_id == 0)
        tcgen05_alloc_1sm(static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_address)), kN);
    __syncthreads();

    const uint32_t tmem = static_cast<uint32_t>(tmem_address);
    const uint64_t a_descriptor = make_smem_descriptor(A);
    const uint64_t b_descriptor = make_smem_descriptor(B);
    constexpr uint32_t descriptor = make_umma_descriptor<kM1sm, kN>();
    if (leader) {
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        uint32_t parity = 0;
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            issue_one_umma_1sm(tmem, a_descriptor, b_descriptor, descriptor, 0);
#pragma unroll
            for (int depth = 1; depth < kDepth; ++depth)
                issue_one_umma_1sm(tmem, a_descriptor, b_descriptor, descriptor, 1);
            commit_umma_1sm(address);
            while (!cuda::ptx::mbarrier_try_wait_parity(&barrier, parity)) {}
            parity ^= 1u;
        }
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();
    float* result = output + static_cast<int64_t>(block) * kM1sm * kN;
    const int row = warp_id * 32 + tid % 32;
#pragma unroll
    for (int fragment = 0; fragment < kN / 32; ++fragment) {
        uint32_t values[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem, warp_id, fragment), values);
        tcgen05_wait_ld();
#pragma unroll
        for (int index = 0; index < 32; ++index)
            result[row * kN + fragment * 32 + index] = __uint_as_float(values[index]);
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_1sm(tmem, kN);
        tcgen05_relinquish_alloc_permit_1sm();
    }
    if (tid == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))) : "memory");
}

extern "C" __global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void umma_2sm_scaling_m256n256k16_d256(int64_t iterations, RunMode mode,
                                        float* __restrict__ output,
                                        unsigned int* __restrict__ arrivals,
                                        int* __restrict__ resident,
                                        int* __restrict__ smids) {
    const int tid = threadIdx.x;
    const int block = static_cast<int>(blockIdx.x);
    if (mode == RunMode::kResidency) {
        residency_handshake(arrivals, resident, smids);
        return;
    }

    const int rank = static_cast<int>(cuda::ptx::get_sreg_cluster_ctarank());
    const int cluster = block / kClusterCtas;
    constexpr int kNLocal = kN / kClusterCtas;
    constexpr int kABytes = kMLocal * kK * 2;
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);
    const int warp_id = tid / 32;

    for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) {
        const int row = idx / kK;
        const int k = idx % kK;
        const int value = ((rank * kMLocal + row + 3 * k) % 7) - 3;
        A[smem_core_tile_index(row / 8, row % 8, k)] = __float2bfloat16(static_cast<float>(value));
    }
    for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) {
        const int column = idx / kK;
        const int k = idx % kK;
        const int value = ((2 * k + rank * kNLocal + column) % 5) - 2;
        B[smem_core_tile_index(column / 8, column % 8, k)] =
            __float2bfloat16(static_cast<float>(value));
    }

    __shared__ uint64_t barrier;
    __shared__ int tmem_address;
    __syncthreads();
    if (tid == 0) {
        cuda::ptx::mbarrier_init(&barrier, 1u);
        fence_mbarrier_init_release_cluster();
        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
    }
    __syncthreads();

    bool leader = false;
    if (tid < 32) leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();
    if (warp_id == 0)
        tcgen05_alloc_2sm(static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_address)), kN);
    __syncthreads();
    const uint32_t tmem = static_cast<uint32_t>(tmem_address);
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    const uint64_t a_descriptor = make_smem_descriptor(A);
    const uint64_t b_descriptor = make_smem_descriptor(B);
    constexpr uint32_t descriptor = make_umma_descriptor<kM2sm, kN>();
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    uint32_t parity = 0;
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (leader) {
            if (rank == 0) {
                issue_one_umma_2sm(tmem, a_descriptor, b_descriptor, descriptor, 0);
#pragma unroll
                for (int depth = 1; depth < kDepth; ++depth)
                    issue_one_umma_2sm(tmem, a_descriptor, b_descriptor, descriptor, 1);
                commit_umma_2sm_multicast(address, 0x0003u);
            }
            while (!cuda::ptx::mbarrier_try_wait_parity(&barrier, parity)) {}
            parity ^= 1u;
        }
        __syncthreads();
        cuda::ptx::barrier_cluster_arrive();
        cuda::ptx::barrier_cluster_wait();
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();
    float* result = output + static_cast<int64_t>(cluster) * kM2sm * kN;
    const int row = rank * kMLocal + warp_id * 32 + tid % 32;
#pragma unroll
    for (int fragment = 0; fragment < kN / 32; ++fragment) {
        uint32_t values[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem, warp_id, fragment), values);
        tcgen05_wait_ld();
#pragma unroll
        for (int index = 0; index < 32; ++index)
            result[static_cast<int64_t>(row) * kN + fragment * 32 + index] =
                __uint_as_float(values[index]);
    }

    __syncthreads();
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();
    if (warp_id == 0) {
        tcgen05_dealloc_2sm(tmem, kN);
        tcgen05_relinquish_alloc_permit_2sm();
    }
    if (tid == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&barrier))) : "memory");
}

namespace {

using ScalingKernel = void (*)(int64_t, RunMode, float*, unsigned int*, int*, int*);

struct Plan {
    const char* method;
    const char* scale;
    int m;
    int group;
    int blocks;
    int work_units;
    int observed_sms = 0;
    int64_t total_flops = 0;
    ScalingKernel kernel;
    float* output = nullptr;
    unsigned int* arrivals = nullptr;
    int* resident = nullptr;
    int* smids = nullptr;
};

void launch(const Plan& plan, int64_t iterations, RunMode mode, size_t reservation) {
    plan.kernel<<<plan.blocks, kThreadsPerCta, reservation>>>(
        iterations, mode, plan.output, plan.arrivals, plan.resident, plan.smids);
    CUDA_CHECK_FATAL(cudaGetLastError());
}

void initialize(Plan& plan, int64_t iterations) {
    plan.work_units = plan.blocks / plan.group;
    plan.total_flops = 2 * static_cast<int64_t>(plan.m) * kN * kK *
                       kDepth * iterations * plan.work_units;
    CUDA_CHECK_FATAL(cudaMalloc(&plan.output,
        static_cast<size_t>(plan.work_units) * plan.m * kN * sizeof(float)));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.arrivals, sizeof(unsigned int)));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.resident, static_cast<size_t>(plan.blocks) * sizeof(int)));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.smids, static_cast<size_t>(plan.blocks) * sizeof(int)));
}

void probe_residency(Plan& plan, size_t reservation) {
    CUDA_CHECK_FATAL(cudaMemset(plan.arrivals, 0, sizeof(unsigned int)));
    CUDA_CHECK_FATAL(cudaMemset(plan.resident, 0, static_cast<size_t>(plan.blocks) * sizeof(int)));
    CUDA_CHECK_FATAL(cudaMemset(plan.smids, 0xFF, static_cast<size_t>(plan.blocks) * sizeof(int)));
    launch(plan, 1, RunMode::kResidency, reservation);
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    std::vector<int> resident(static_cast<size_t>(plan.blocks));
    std::vector<int> smids(static_cast<size_t>(plan.blocks));
    CUDA_CHECK_FATAL(cudaMemcpy(resident.data(), plan.resident,
                                resident.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FATAL(cudaMemcpy(smids.data(), plan.smids,
                                smids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (std::any_of(resident.begin(), resident.end(), [](int value) { return value != 1; }))
        fail("%s/%s: blocks are not simultaneously resident", plan.method, plan.scale);
    std::sort(smids.begin(), smids.end());
    plan.observed_sms = static_cast<int>(std::unique(smids.begin(), smids.end()) - smids.begin());
    if (plan.observed_sms != plan.blocks)
        fail("%s/%s: observed %d of %d planned SMs", plan.method, plan.scale,
             plan.observed_sms, plan.blocks);
}

std::vector<float> build_reference() {
    std::vector<float> reference(static_cast<size_t>(kM2sm) * kN);
    for (int row = 0; row < kM2sm; ++row) {
        for (int column = 0; column < kN; ++column) {
            int value = 0;
            for (int k = 0; k < kK; ++k)
                value += (((row + 3 * k) % 7) - 3) * (((2 * k + column) % 5) - 2);
            reference[static_cast<size_t>(row) * kN + column] =
                static_cast<float>(value * kDepth);
        }
    }
    return reference;
}

void validate(const Plan& plan, int64_t iterations, size_t reservation,
              const std::vector<float>& reference) {
    launch(plan, iterations, RunMode::kCompute, reservation);
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    std::vector<float> output(static_cast<size_t>(plan.work_units) * plan.m * kN);
    CUDA_CHECK_FATAL(cudaMemcpy(output.data(), plan.output, output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    const size_t unit_elements = static_cast<size_t>(plan.m) * kN;
    for (int unit = 0; unit < plan.work_units; ++unit)
        for (size_t index = 0; index < unit_elements; ++index)
            if (output[static_cast<size_t>(unit) * unit_elements + index] != reference[index])
                fail("%s/%s: numerical validation failed", plan.method, plan.scale);
}

size_t choose_reservation(const cudaDeviceProp& properties) {
    // Reserve enough shared memory to limit occupancy to one CTA per SM.
    size_t reservation = (properties.sharedMemPerMultiprocessor / 2 + 1024 + 127) / 128 * 128;
    reservation = std::min(reservation, static_cast<size_t>(properties.sharedMemPerBlockOptin));
    if (reservation < static_cast<size_t>((kMLocal + kN) * kK * 2))
        fail("insufficient shared memory");
    return reservation;
}

int active_clusters(size_t reservation, int blocks) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = kClusterCtas;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned int>(blocks), 1, 1);
    config.blockDim = dim3(kThreadsPerCta, 1, 1);
    config.dynamicSmemBytes = reservation;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    CUDA_CHECK_FATAL(cudaOccupancyMaxActiveClusters(
        &clusters, reinterpret_cast<const void*>(&umma_2sm_scaling_m256n256k16_d256), &config));
    return clusters;
}

void verify_occupancy(ScalingKernel kernel, size_t reservation) {
    int occupancy = 0;
    CUDA_CHECK_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, reinterpret_cast<const void*>(kernel), kThreadsPerCta, reservation));
    if (occupancy != 1) fail("device-scale UMMA requires one CTA per SM");
}

void print_row(const Plan& plan, int hardware_sms, int64_t sample,
               int64_t iterations, float milliseconds) {
    const double throughput = plan.total_flops / static_cast<double>(milliseconds) / 1e9;
    const int clusters = plan.group == 2 ? plan.work_units : 0;
    std::printf("%s,%s,%lld,%d,%lld,%d,%d,%d,%d,%d,"
                "all_blocks_simultaneously_resident,%.6f,%.6f,%.6f,OK\n",
                plan.method, plan.scale, static_cast<long long>(sample), plan.group,
                static_cast<long long>(iterations), plan.work_units, clusters, hardware_sms,
                plan.blocks, plan.observed_sms, static_cast<double>(milliseconds),
                throughput, throughput / plan.blocks);
}

void release(Plan& plan) {
    CUDA_CHECK_FATAL(cudaFree(plan.output));
    CUDA_CHECK_FATAL(cudaFree(plan.arrivals));
    CUDA_CHECK_FATAL(cudaFree(plan.resident));
    CUDA_CHECK_FATAL(cudaFree(plan.smids));
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig config;
    std::string error;
    if (!parse_cli(argc, argv, &config, &error, true)) {
        std::fprintf(stderr, "umma_device_scaling: %s\n", error.c_str());
        print_usage(stderr);
        return 2;
    }
    if (config.help) { print_usage(stdout); return 0; }

    const cudaDeviceProp properties = benchmark_device_properties();
    const int sms = properties.multiProcessorCount;
    if (sms < 2 || sms % 2) fail("whole-device comparison requires an even SM count");
    const size_t reservation = choose_reservation(properties);
    CUDA_CHECK_FATAL(cudaFuncSetAttribute(&umma_1sm_scaling_m128n256k16_d256,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(reservation)));
    CUDA_CHECK_FATAL(cudaFuncSetAttribute(&umma_2sm_scaling_m256n256k16_d256,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(reservation)));
    verify_occupancy(&umma_1sm_scaling_m128n256k16_d256, reservation);
    verify_occupancy(&umma_2sm_scaling_m256n256k16_d256, reservation);
    if (active_clusters(reservation, sms) < sms / kClusterCtas)
        fail("the GPU cannot host every required two-CTA cluster");

    // Compare one isolated work unit with all SMs or all two-CTA clusters.
    std::vector<Plan> plans = {
        {"umma_1sm", "isolated", kM1sm, 1, 1, 1, 0, 0,
         &umma_1sm_scaling_m128n256k16_d256},
        {"umma_2sm", "isolated", kM2sm, 2, 2, 1, 0, 0,
         &umma_2sm_scaling_m256n256k16_d256},
        {"umma_1sm", "device_scale", kM1sm, 1, sms, sms, 0, 0,
         &umma_1sm_scaling_m128n256k16_d256},
        {"umma_2sm", "device_scale", kM2sm, 2, sms, sms / 2, 0, 0,
         &umma_2sm_scaling_m256n256k16_d256},
    };
    const std::vector<float> reference = build_reference();
    for (Plan& plan : plans) {
        initialize(plan, config.iterations);
        probe_residency(plan, reservation);
        validate(plan, config.iterations, reservation, reference);
    }

    for (Plan& plan : plans) {
        for (int64_t index = 0; index < config.warmup_iterations; ++index) {
            launch(plan, config.iterations, RunMode::kCompute, reservation);
            CUDA_CHECK_FATAL(cudaDeviceSynchronize());
        }
    }

    cudaEvent_t started = nullptr, stopped = nullptr;
    CUDA_CHECK_FATAL(cudaEventCreate(&started));
    CUDA_CHECK_FATAL(cudaEventCreate(&stopped));
    std::puts("method,scale,sample_index,cta_group,iterations,work_unit_count,cluster_count,"
              "hardware_sm_count,planned_active_sm_count,observed_unique_sm_count,"
              "residency_evidence,kernel_time_ms,total_tflops,tflops_per_planned_active_sm,"
              "correctness");
    for (int64_t sample = 0; sample < config.repetitions; ++sample) {
        // Alternate measurement order to reduce systematic timing drift.
        for (size_t step = 0; step < plans.size(); ++step) {
            Plan& plan = plans[sample % 2 == 0 ? step : plans.size() - 1 - step];
            CUDA_CHECK_FATAL(cudaEventRecord(started));
            launch(plan, config.iterations, RunMode::kCompute, reservation);
            CUDA_CHECK_FATAL(cudaEventRecord(stopped));
            CUDA_CHECK_FATAL(cudaEventSynchronize(stopped));
            float milliseconds = 0;
            CUDA_CHECK_FATAL(cudaEventElapsedTime(&milliseconds, started, stopped));
            if (!(milliseconds > 0)) fail("kernel duration must be positive");
            print_row(plan, sms, sample, config.iterations, milliseconds);
        }
    }

    for (Plan& plan : plans) release(plan);
    CUDA_CHECK_FATAL(cudaEventDestroy(started));
    CUDA_CHECK_FATAL(cudaEventDestroy(stopped));
    return 0;
}
