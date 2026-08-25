// Single-SM BF16 UMMA throughput measured with %clock64.

#define UMMA_METHOD "umma_1sm"
#include "umma_common.cuh"

namespace {

constexpr int kM = 128;

template <int N, int DEPTH>
__device__ __forceinline__ void umma_1sm_body(int64_t iterations, TimingMode timing_mode,
                                               float* __restrict__ g_d_out,
                                               unsigned long long* __restrict__ g_elapsed_cycles) {
    const int tid = threadIdx.x;
    constexpr int kABytes = kM * kK * 2;

    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);
    const int warp_id = tid / 32;

    for (int idx = tid; idx < kM * kK; idx += kThreadsPerCta) {
        const int row = idx / kK;
        const int k = idx % kK;
        const int value = ((row + 3 * k) % 7) - 3;
        A[smem_core_tile_index(row / 8, row % 8, k)] = __float2bfloat16(static_cast<float>(value));
    }
    for (int idx = tid; idx < N * kK; idx += kThreadsPerCta) {
        const int col = idx / kK;
        const int k = idx % kK;
        const int value = ((2 * k + col) % 5) - 2;
        B[smem_core_tile_index(col / 8, col % 8, k)] = __float2bfloat16(static_cast<float>(value));
    }

    __shared__ uint64_t mbar;
    __shared__ int tmem_addr_shared;

    __syncthreads();
    if (tid == 0) {
        cuda::ptx::mbarrier_init(&mbar, 1u);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    bool is_leader = false;
    if (tid < 32) is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);

    if (warp_id == 0) {
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared));
        tcgen05_alloc_1sm(address, static_cast<uint32_t>(N));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);
    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_umma_descriptor<kM, N>();

    unsigned long long elapsed_cycles = 0;
    if (is_leader) {
        // Time only the elected thread's UMMA issue-and-completion loop.
        const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        uint64_t start_clock = 0, end_clock = 0;
        uint32_t parity = 0;
        if (timing_mode == TimingMode::kTimed)
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");

        for (int64_t it = 0; it < iterations; ++it) {
            issue_one_umma_1sm(tmem_d, a_desc, b_desc, idesc, 0);
#pragma unroll
            for (int depth = 1; depth < DEPTH; ++depth)
                issue_one_umma_1sm(tmem_d, a_desc, b_desc, idesc, 1);
            commit_umma_1sm(mbar_addr);
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {}
            parity ^= 1u;
        }
        if (timing_mode == TimingMode::kTimed) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
            elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
        }
        g_elapsed_cycles[0] = elapsed_cycles;
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();
    constexpr int kFragments = N / 32;
    const int lane = tid % 32;
    const int row = warp_id * 32 + lane;
#pragma unroll
    for (int fragment = 0; fragment < kFragments; ++fragment) {
        uint32_t values[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, fragment), values);
        tcgen05_wait_ld();
#pragma unroll
        for (int index = 0; index < 32; ++index)
            g_d_out[row * N + fragment * 32 + index] = __uint_as_float(values[index]);
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_1sm(tmem_d, static_cast<uint32_t>(N));
        tcgen05_relinquish_alloc_permit_1sm();
    }
    if (tid == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))) : "memory");
}

}  // namespace

#define UMMA_KERNEL(N, DEPTH)                                                       \
    extern "C" __global__ __launch_bounds__(128)                                    \
    void umma_1sm_m128n##N##k16_d##DEPTH(int64_t iterations, TimingMode mode,        \
                                         float* output, unsigned long long* cycles) {\
        umma_1sm_body<N, DEPTH>(iterations, mode, output, cycles);                   \
    }
UMMA_KERNEL(64, 4) UMMA_KERNEL(64, 16) UMMA_KERNEL(64, 64) UMMA_KERNEL(64, 256)
UMMA_KERNEL(128, 4) UMMA_KERNEL(128, 16) UMMA_KERNEL(128, 64) UMMA_KERNEL(128, 256)
UMMA_KERNEL(256, 4) UMMA_KERNEL(256, 16) UMMA_KERNEL(256, 64) UMMA_KERNEL(256, 256)
#undef UMMA_KERNEL

namespace {

#define UMMA_SPEC(N, DEPTH) {N, DEPTH, &umma_1sm_m128n##N##k16_d##DEPTH}
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
        std::fprintf(stderr, "umma_1sm: %s\n", error.c_str());
        print_usage(stderr);
        return 2;
    }
    if (config.help) { print_usage(stdout); return 0; }
    benchmark_device_properties();
    for (const auto& specialization : kSpecializations)
        if (specialization.n == config.n && specialization.depth == config.depth)
            return run_isolated(specialization, config, kM, 1,
                                kM * kK * 2 + config.n * kK * 2);
    fail("unsupported configuration");
}
