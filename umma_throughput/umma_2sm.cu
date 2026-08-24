// Two-SM BF16 UMMA throughput measured with the leader's %clock64.

#define UMMA_METHOD "umma_2sm"
#include "umma_common.cuh"

namespace {

constexpr int kMGlobal = 256;
constexpr int kMLocal = 128;

template <int N, int DEPTH>
__device__ __forceinline__ void umma_2sm_body(int64_t iterations, TimingMode timing_mode,
                                               float* __restrict__ g_d_out,
                                               unsigned long long* __restrict__ g_elapsed_cycles) {
    const int tid = threadIdx.x;
    const int cta_rank = static_cast<int>(cuda::ptx::get_sreg_cluster_ctarank());
    constexpr int kNLocal = N / kClusterCtas;
    constexpr int kABytes = kMLocal * kK * 2;

    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);
    const int warp_id = tid / 32;

    for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) {
        const int local_row = idx / kK;
        const int k = idx % kK;
        const int value = ((cta_rank * kMLocal + local_row + 3 * k) % 7) - 3;
        A[smem_core_tile_index(local_row / 8, local_row % 8, k)] =
            __float2bfloat16(static_cast<float>(value));
    }
    for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) {
        const int local_col = idx / kK;
        const int k = idx % kK;
        const int value = ((2 * k + cta_rank * kNLocal + local_col) % 5) - 2;
        B[smem_core_tile_index(local_col / 8, local_col % 8, k)] =
            __float2bfloat16(static_cast<float>(value));
    }

    __shared__ uint64_t mbar;
    __shared__ int tmem_addr_shared;

    __syncthreads();
    if (tid == 0) {
        cuda::ptx::mbarrier_init(&mbar, 1u);
        fence_mbarrier_init_release_cluster();
        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
    }
    __syncthreads();

    bool is_leader = false;
    if (tid < 32) is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    if (warp_id == 0) {
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared));
        tcgen05_alloc_2sm(address, static_cast<uint32_t>(N));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_umma_descriptor<kMGlobal, N>();
    const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    unsigned long long elapsed_cycles = 0;
    uint64_t start_clock = 0, end_clock = 0;
    if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");

    uint32_t parity = 0;
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (is_leader) {
            if (cta_rank == 0) {
                issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0);
#pragma unroll
                for (int depth = 1; depth < DEPTH; ++depth)
                    issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 1);
                commit_umma_2sm_multicast(mbar_addr, 0x0003u);
            }
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {}
            parity ^= 1u;
        }
        __syncthreads();
        cuda::ptx::barrier_cluster_arrive();
        cuda::ptx::barrier_cluster_wait();
    }

    if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
        elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
    }
    if (is_leader && cta_rank == 0) g_elapsed_cycles[0] = elapsed_cycles;

    __syncthreads();
    tcgen05_fence_after_thread_sync();
    constexpr int kFragments = N / 32;
    const int local_row = warp_id * 32 + tid % 32;
    const int global_row = cta_rank * kMLocal + local_row;
#pragma unroll
    for (int fragment = 0; fragment < kFragments; ++fragment) {
        uint32_t values[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, fragment), values);
        tcgen05_wait_ld();
#pragma unroll
        for (int index = 0; index < 32; ++index)
            g_d_out[static_cast<int64_t>(global_row) * N + fragment * 32 + index] =
                __uint_as_float(values[index]);
    }

    __syncthreads();
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();
    if (warp_id == 0) {
        tcgen05_dealloc_2sm(tmem_d, static_cast<uint32_t>(N));
        tcgen05_relinquish_alloc_permit_2sm();
    }
    if (tid == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))) : "memory");
}

}  // namespace

#define UMMA_KERNEL(N, DEPTH)                                                        \
    extern "C" __global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)           \
    void umma_2sm_m256n##N##k16_d##DEPTH(int64_t iterations, TimingMode mode,         \
                                         float* output, unsigned long long* cycles) { \
        umma_2sm_body<N, DEPTH>(iterations, mode, output, cycles);                    \
    }
UMMA_KERNEL(64, 4) UMMA_KERNEL(64, 16) UMMA_KERNEL(64, 64) UMMA_KERNEL(64, 256)
UMMA_KERNEL(128, 4) UMMA_KERNEL(128, 16) UMMA_KERNEL(128, 64) UMMA_KERNEL(128, 256)
UMMA_KERNEL(256, 4) UMMA_KERNEL(256, 16) UMMA_KERNEL(256, 64) UMMA_KERNEL(256, 256)
#undef UMMA_KERNEL

namespace {

#define UMMA_SPEC(N, DEPTH) {N, DEPTH, &umma_2sm_m256n##N##k16_d##DEPTH}
constexpr IsolatedSpecialization kSpecializations[] = {
    UMMA_SPEC(64, 4), UMMA_SPEC(64, 16), UMMA_SPEC(64, 64), UMMA_SPEC(64, 256),
    UMMA_SPEC(128, 4), UMMA_SPEC(128, 16), UMMA_SPEC(128, 64), UMMA_SPEC(128, 256),
    UMMA_SPEC(256, 4), UMMA_SPEC(256, 16), UMMA_SPEC(256, 64), UMMA_SPEC(256, 256),
};
#undef UMMA_SPEC

}  // namespace

int main(int argc, char** argv) {
    CliConfig config;
    std::string error;
    if (!parse_cli(argc, argv, &config, &error)) {
        std::fprintf(stderr, "umma_2sm: %s\n", error.c_str());
        print_usage(stderr);
        return 2;
    }
    if (config.help) { print_usage(stdout); return 0; }
    benchmark_device_properties();
    for (const auto& specialization : kSpecializations)
        if (specialization.n == config.n && specialization.depth == config.depth)
            return run_isolated(specialization, config, kMGlobal, kClusterCtas,
                                kMLocal * kK * 2 + config.n / kClusterCtas * kK * 2);
    fail("unsupported configuration");
}
