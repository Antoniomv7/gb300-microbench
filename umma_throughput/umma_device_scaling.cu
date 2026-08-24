// Compare isolated and whole-device UMMA using whole-kernel CUDA events.

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
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
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>

#include "../benchmark_common.cuh"

namespace {

constexpr int kThreadsPerCta = 128;
constexpr int kN = 256;              // frozen: the ceiling-candidate N
constexpr int kDepth = 256;          // frozen: the ceiling-candidate depth
constexpr int kK = 16;               // implied by .kind::f16 dense BF16
constexpr int kMLocal = 128;         // rows of A/D owned by one CTA
constexpr int kM1sm = 128;           // joint M of a cta_group::1 work unit
constexpr int kM2sm = 256;           // joint M of a cta_group::2 work unit
constexpr int kClusterCtas = 2;
constexpr const char* kSchemaVersion = "umma-device-scaling.v1";
constexpr const char* kOperandPath = "smem_smem";
constexpr const char* kInputType = "bf16";
constexpr const char* kAccumulatorType = "fp32";
constexpr const char* kNotApplicable = "not_applicable";

constexpr unsigned long long kResidencyTimeoutCycles = 2000000000ULL;

enum class RunMode : int32_t { kUntimed = 0, kTimed = 1, kResidency = 2 };

[[noreturn]] void fail(const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "umma_device_scaling: ERROR: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

#define CUDA_CHECK_FATAL(call)                                                   \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fail("cuda_error=%s detail=\"%s\" at %s:%d", cudaGetErrorName(err_), \
                 cudaGetErrorString(err_), __FILE__, __LINE__);                  \
        }                                                                        \
    } while (0)

int64_t checked_mul_i64(int64_t a, int64_t b, const char* what) {
    const __int128 result = static_cast<__int128>(a) * static_cast<__int128>(b);
    if (result > static_cast<__int128>(INT64_MAX) || result < static_cast<__int128>(INT64_MIN)) {
        fail("integer overflow computing %s (a=%lld b=%lld)", what, (long long)a, (long long)b);
    }
    return static_cast<int64_t>(result);
}

size_t checked_alloc_bytes(int64_t count, size_t element_bytes, const char* what) {
    if (count <= 0) fail("invalid element count %lld for %s", (long long)count, what);
    const __int128 total = static_cast<__int128>(count) * static_cast<__int128>(element_bytes);
    if (total > static_cast<__int128>(SIZE_MAX)) fail("allocation size overflow for %s", what);
    return static_cast<size_t>(total);
}

__device__ __forceinline__ int smem_core_tile_index(int group_idx, int pos_in_group, int k) {
    constexpr int kSboElem = 128;
    constexpr int kLboElem = 64;
    constexpr int kT = 8;
    const int chunk = k / kT;
    const int t = k % kT;
    return group_idx * kSboElem + chunk * kLboElem + pos_in_group * kT + t;
}

__device__ __forceinline__ uint64_t make_smem_descriptor(const void* smem_ptr) {
    constexpr uint32_t kLboBytes = 128;
    constexpr uint32_t kSboBytes = 256;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFFu);                    // bits 0-13
    desc |= static_cast<uint64_t>((kLboBytes >> 4) & 0x3FFFu) << 16;         // bits 16-29
    desc |= static_cast<uint64_t>((kSboBytes >> 4) & 0x3FFFu) << 32;         // bits 32-45
    desc |= (UINT64_C(1) << 46);                                             // bits 46-48 = 0b001
    return desc;
}

template <int M>
__device__ constexpr uint32_t make_instruction_descriptor() {
    static_assert(M == kM1sm || M == kM2sm, "M must be 128 (1-SM) or 256 (2-SM)");
    static_assert(((static_cast<uint32_t>(kN) >> 3) & ~0x3Fu) == 0, "N field overflows bits 17-22");
    static_assert(((static_cast<uint32_t>(M) >> 4) & ~0x1Fu) == 0, "M field overflows bits 24-28");
    uint32_t desc = 0;
    desc |= (1u << 4);                                    // dtype = FP32
    desc |= (1u << 7);                                    // atype = BF16
    desc |= (1u << 10);                                   // btype = BF16
    desc |= ((static_cast<uint32_t>(kN) >> 3) << 17);     // N >> 3
    desc |= ((static_cast<uint32_t>(M) >> 4) << 24);      // M >> 4
    return desc;
}

template <int M>
__device__ constexpr bool validate_instruction_descriptor() {
    constexpr uint32_t d = make_instruction_descriptor<M>();
    return (d & 0x3u) == 0 && ((d >> 2) & 0x1u) == 0 && ((d >> 3) & 0x1u) == 0 &&
           ((d >> 4) & 0x3u) == 1 && ((d >> 6) & 0x1u) == 0 && ((d >> 7) & 0x7u) == 1 &&
           ((d >> 10) & 0x7u) == 1 && ((d >> 13) & 0x1u) == 0 && ((d >> 14) & 0x1u) == 0 &&
           ((d >> 15) & 0x1u) == 0 && ((d >> 16) & 0x1u) == 0 &&
           ((d >> 17) & 0x3Fu) == (static_cast<uint32_t>(kN) >> 3) && ((d >> 23) & 0x1u) == 0 &&
           ((d >> 24) & 0x1Fu) == (static_cast<uint32_t>(M) >> 4) && ((d >> 29) & 0x1u) == 0 &&
           ((d >> 30) & 0x3u) == 0;
}

static_assert(validate_instruction_descriptor<kM1sm>(), "1-SM instruction descriptor is malformed");
static_assert(validate_instruction_descriptor<kM2sm>(), "2-SM instruction descriptor is malformed");

__device__ __forceinline__ void issue_one_umma_1sm(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                   uint32_t idesc, int enable_input_d) {
    asm volatile(
        "{\n\t.reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t}\n\t"
        :
        : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
        : "memory");
}

__device__ __forceinline__ void issue_one_umma_2sm(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                   uint32_t idesc, int enable_input_d) {
    asm volatile(
        "{\n\t.reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t}\n\t"
        :
        : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
        : "memory");
}

__device__ __forceinline__ void commit_umma_1sm(uint32_t mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 : : "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void commit_umma_2sm_multicast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        : : "r"(mbar_addr), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tcgen05_alloc_1sm(uint32_t dst_smem_addr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(dst_smem_addr), "r"(n_cols) : "memory");
}

__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t dst_smem_addr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(dst_smem_addr), "r"(n_cols) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_1sm(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 : : "r"(taddr), "r"(n_cols) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 : : "r"(taddr), "r"(n_cols) : "memory");
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

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t out[32]) {
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
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

constexpr uint32_t kTmemLaneShift = 16;
constexpr uint32_t kTmemRowsPerWarp = 32;
constexpr uint32_t kTmemColsPerFragment = 32;

__device__ __forceinline__ uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {
    const uint32_t lane = (static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift;
    return tmem_base + lane + static_cast<uint32_t>(frag) * kTmemColsPerFragment;
}

__device__ __forceinline__ int current_smid() {
    unsigned int smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return static_cast<int>(smid);
}

// All blocks must overlap before full-device coverage is accepted.
__device__ __forceinline__ void residency_handshake(unsigned int* __restrict__ g_arrival,
                                                    int* __restrict__ g_resident_ok) {
    if (threadIdx.x != 0) return;
    const unsigned int expected = gridDim.x;
    atomicAdd(g_arrival, 1u);
    const unsigned long long deadline_start = clock64();
    int observed_all = 0;
    while (true) {
        if (atomicAdd(g_arrival, 0u) >= expected) {
            observed_all = 1;
            break;
        }
        if (static_cast<unsigned long long>(clock64() - deadline_start) > kResidencyTimeoutCycles) {
            break;
        }
    }
    g_resident_ok[blockIdx.x] = observed_all;
}

}  // namespace

extern "C" __global__ __launch_bounds__(128) void umma_1sm_scaling_m128n256k16_d256(
    int64_t iterations, RunMode mode, float* __restrict__ g_d_out,
    unsigned long long* __restrict__ g_cycles, int* __restrict__ g_launch_ok,
    unsigned int* __restrict__ g_arrival, int* __restrict__ g_resident_ok,
    int* __restrict__ g_smid) {
    const int tid = threadIdx.x;
    const int block = static_cast<int>(blockIdx.x);

    const bool contract_ok = gridDim.x >= 1 && gridDim.y == 1 && gridDim.z == 1 &&
                             blockDim.x == kThreadsPerCta && blockDim.y == 1 && blockDim.z == 1;
    if (!contract_ok) {
        if (tid == 0) g_launch_ok[block] = 0;
        return;
    }
    if (tid == 0) {
        g_launch_ok[block] = 1;
        g_smid[block] = current_smid();
    }

    if (mode == RunMode::kResidency) {
        residency_handshake(g_arrival, g_resident_ok);
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
        tcgen05_alloc_1sm(static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared)),
                          static_cast<uint32_t>(kN));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_instruction_descriptor<kM1sm>();

    unsigned long long elapsed_cycles = 0;
    if (is_leader) {
        const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        uint64_t start_clock = 0, end_clock = 0;
        uint32_t parity = 0;
        if (mode == RunMode::kTimed) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");
        }
        for (int64_t it = 0; it < iterations; ++it) {
            issue_one_umma_1sm(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/0);
#pragma unroll
            for (int m = 1; m < kDepth; ++m) {
                issue_one_umma_1sm(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/1);
            }
            commit_umma_1sm(mbar_addr);
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {
            }
            parity ^= 1u;
        }
        if (mode == RunMode::kTimed) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
            elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
        }
        g_cycles[block] = elapsed_cycles;
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();

    constexpr int kFragments = kN / 32;
    float* d_out = g_d_out + static_cast<int64_t>(block) * (kM1sm * kN);
    const int lane = tid % 32;
    const int row = warp_id * 32 + lane;
#pragma unroll
    for (int frag = 0; frag < kFragments; ++frag) {
        uint32_t regs[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);
        tcgen05_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            d_out[row * kN + frag * 32 + i] = __uint_as_float(regs[i]);
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc_1sm(tmem_d, static_cast<uint32_t>(kN));
        tcgen05_relinquish_alloc_permit_1sm();
    }
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))) : "memory");
    }
}

extern "C" __global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void umma_2sm_scaling_m256n256k16_d256(
    int64_t iterations, RunMode mode, float* __restrict__ g_d_out,
    unsigned long long* __restrict__ g_cycles, int* __restrict__ g_launch_ok,
    unsigned int* __restrict__ g_arrival, int* __restrict__ g_resident_ok,
    int* __restrict__ g_smid) {
    const int tid = threadIdx.x;
    const int block = static_cast<int>(blockIdx.x);
    const uint32_t cluster_ctarank = cuda::ptx::get_sreg_cluster_ctarank();
    const uint32_t cluster_nctarank = cuda::ptx::get_sreg_cluster_nctarank();

    const bool contract_ok = gridDim.x >= kClusterCtas && gridDim.x % kClusterCtas == 0 &&
                             gridDim.y == 1 && gridDim.z == 1 && blockDim.x == kThreadsPerCta &&
                             blockDim.y == 1 && blockDim.z == 1 &&
                             cluster_nctarank == kClusterCtas &&
                             cluster_ctarank == static_cast<uint32_t>(block % kClusterCtas);
    if (!contract_ok) {
        if (tid == 0) g_launch_ok[block] = 0;
        return;
    }
    if (tid == 0) {
        g_launch_ok[block] = 1;
        g_smid[block] = current_smid();
    }

    if (mode == RunMode::kResidency) {
        residency_handshake(g_arrival, g_resident_ok);
        return;
    }

    const int cta_rank = static_cast<int>(cluster_ctarank);
    const int cluster_index = block / kClusterCtas;

    constexpr int kNLocal = kN / kClusterCtas;
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
        tcgen05_alloc_2sm(static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared)),
                          static_cast<uint32_t>(kN));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);

    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_instruction_descriptor<kM2sm>();
    const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    unsigned long long elapsed_cycles = 0;
    uint64_t start_clock = 0, end_clock = 0;
    if (is_leader && cta_rank == 0 && mode == RunMode::kTimed) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");
    }

    uint32_t parity = 0;
    for (int64_t it = 0; it < iterations; ++it) {
        if (is_leader) {
            if (cta_rank == 0) {
                issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/0);
#pragma unroll
                for (int m = 1; m < kDepth; ++m) {
                    issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/1);
                }
                commit_umma_2sm_multicast(mbar_addr, /*cta_mask=*/0x0003u);
            }
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {
            }
            parity ^= 1u;
        }
        __syncthreads();
        cuda::ptx::barrier_cluster_arrive();
        cuda::ptx::barrier_cluster_wait();
    }

    if (is_leader && cta_rank == 0 && mode == RunMode::kTimed) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
        elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
    }
    if (is_leader && cta_rank == 0) {
        g_cycles[cluster_index] = elapsed_cycles;
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();

    constexpr int kFragments = kN / 32;
    float* d_out = g_d_out + static_cast<int64_t>(cluster_index) * (kM2sm * kN);
    const int lane = tid % 32;
    const int global_row = cta_rank * kMLocal + warp_id * 32 + lane;
#pragma unroll
    for (int frag = 0; frag < kFragments; ++frag) {
        uint32_t regs[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);
        tcgen05_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            d_out[static_cast<int64_t>(global_row) * kN + frag * 32 + i] = __uint_as_float(regs[i]);
        }
    }

    __syncthreads();
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    if (warp_id == 0) {
        tcgen05_dealloc_2sm(tmem_d, static_cast<uint32_t>(kN));
        tcgen05_relinquish_alloc_permit_2sm();
    }
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     : : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))) : "memory");
    }
}

namespace {


typedef void (*KernelFn)(int64_t, RunMode, float*, unsigned long long*, int*, unsigned int*, int*,
                         int*);

int32_t reference_a(int global_row, int k) { return ((global_row + 3 * k) % 7) - 3; }
int32_t reference_b(int k, int col) { return ((2 * k + col) % 5) - 2; }

std::vector<double> build_reference() {
    std::vector<double> table(static_cast<size_t>(kM2sm) * kN);
    for (int row = 0; row < kM2sm; ++row) {
        for (int col = 0; col < kN; ++col) {
            int64_t sum = 0;
            for (int k = 0; k < kK; ++k) {
                sum += static_cast<int64_t>(reference_a(row, k)) *
                       static_cast<int64_t>(reference_b(k, col));
            }
            table[static_cast<size_t>(row) * kN + col] =
                static_cast<double>(kDepth) * static_cast<double>(sum);
        }
    }
    return table;
}

struct ValidationResult {
    bool ok = false;
    int64_t mismatches = 0;
    int64_t first_mismatch_index = -1;
    double first_mismatch_expected = 0.0;
    double first_mismatch_obtained = 0.0;
    double max_abs_error = 0.0;
};

ValidationResult validate_d(const std::vector<float>& d_out, int work_units, int m,
                            const std::vector<double>& reference) {
    ValidationResult r;
    const size_t unit_elements = static_cast<size_t>(m) * kN;
    for (int unit = 0; unit < work_units; ++unit) {
        const size_t base = static_cast<size_t>(unit) * unit_elements;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < kN; ++col) {
                const size_t offset = static_cast<size_t>(row) * kN + col;
                const double expected = reference[offset];
                const double obtained = static_cast<double>(d_out[base + offset]);
                const double abs_error = std::fabs(obtained - expected);
                if (abs_error > r.max_abs_error) r.max_abs_error = abs_error;
                if (obtained != expected) {
                    if (r.mismatches == 0) {
                        r.first_mismatch_index = static_cast<int64_t>(base + offset);
                        r.first_mismatch_expected = expected;
                        r.first_mismatch_obtained = obtained;
                    }
                    ++r.mismatches;
                }
            }
        }
    }
    r.ok = (r.mismatches == 0);
    return r;
}

struct GpuInfo {
    int multi_processor_count = 0;
    size_t shared_mem_per_multiprocessor = 0;
    int shared_mem_per_block_optin = 0;
};

GpuInfo query_gpu_info() {
    const cudaDeviceProp prop = benchmark_device_properties();
    if (prop.multiProcessorCount < 1) fail("device reports %d SMs", prop.multiProcessorCount);
    if (prop.clusterLaunch == 0) fail("device does not report cluster launch support");
    GpuInfo info;
    info.multi_processor_count = prop.multiProcessorCount;
    info.shared_mem_per_multiprocessor = prop.sharedMemPerMultiprocessor;
    info.shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
    return info;
}

struct Plan {
    std::string method;
    std::string scale;
    int cta_group = 0;
    int m = 0;                       // joint M of one work unit
    int cluster_size = 1;            // CTAs per cluster (1 = no cluster declared)
    int grid_blocks = 0;
    int work_units = 0;
    bool device_scale = false;
    bool has_clusters = false;
    int planned_active_sm = 0;
    int unused_sm = 0;
    int occupancy_blocks_per_sm = 0;
    int max_active_clusters = -1;    // -1 -> not_applicable
    KernelFn kernel = nullptr;

    int64_t flops_per_umma = 0;
    int64_t total_umma = 0;
    int64_t total_flops = 0;

    float* d_out = nullptr;
    unsigned long long* cycles = nullptr;
    int* launch_ok = nullptr;
    int* smid = nullptr;
    int* resident_ok = nullptr;
    unsigned int* arrival = nullptr;
    std::vector<float> host_d;

    bool residency_proven = false;
    int residency_ok_blocks = 0;
};

struct LaunchResult {
    float kernel_time_ms = 0.0f;
    int observed_unique_sm = 0;
    unsigned long long cycles_min = 0;
    unsigned long long cycles_max = 0;
    ValidationResult validation;
};

int unique_count(const std::vector<int>& values) {
    std::vector<int> copy = values;
    std::sort(copy.begin(), copy.end());
    copy.erase(std::unique(copy.begin(), copy.end()), copy.end());
    return static_cast<int>(copy.size());
}

std::string join_ids(const std::vector<int>& values, size_t limit) {
    std::vector<int> copy = values;
    std::sort(copy.begin(), copy.end());
    copy.erase(std::unique(copy.begin(), copy.end()), copy.end());
    std::ostringstream oss;
    for (size_t i = 0; i < copy.size() && i < limit; ++i) {
        if (i) oss << ' ';
        oss << copy[i];
    }
    if (copy.size() > limit) oss << " ...(" << copy.size() << " total)";
    return oss.str();
}

std::string coverage_status_of(const Plan& plan, int observed_unique_sm, int hardware_sm) {
    if (!plan.device_scale) return "isolated_unit";
    if (!plan.residency_proven || observed_unique_sm != plan.planned_active_sm) {
        return "incomplete_coverage";
    }
    if (plan.planned_active_sm == hardware_sm) return "full_device_coverage";
    return "maximum_resident_coverage";
}

bool coverage_is_evidenced(const Plan& plan, int observed_unique_sm) {
    return plan.residency_proven && observed_unique_sm == plan.planned_active_sm;
}

struct CliConfig {
    bool help = false;
    bool has_run_kind = false;
    std::string run_kind;
    std::string campaign_kind = "none";
    bool has_campaign_kind = false;
    bool has_iterations = false;
    int64_t iterations = 0;
    bool has_warmup_iterations = false;
    int64_t warmup_iterations = 0;
    bool has_repetitions = false;
    int64_t repetitions = 0;
};

void print_usage(std::FILE* out) {
    std::fprintf(out,
        "Usage: umma_device_scaling --run-kind {smoke,benchmark} --iterations N\n"
        "       --warmup-iterations N --repetitions N\n"
        "       [--campaign-kind {none,pilot,final}]\n");
}

bool parse_cli(int argc, char** argv, CliConfig* cfg, std::string* err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](void) -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };
        auto once = [&](bool& flag, const char* name) {
            if (flag) { *err = std::string(name) + " specified more than once"; return false; }
            flag = true;
            return true;
        };

        if (arg == "--help") {
            if (!once(cfg->help, "--help")) return false;
            continue;
        }
        if (arg == "--run-kind") {
            if (!once(cfg->has_run_kind, "--run-kind")) return false;
            const auto v = next_value();
            if (!v || (*v != "smoke" && *v != "benchmark")) {
                *err = "--run-kind must be 'smoke' or 'benchmark'";
                return false;
            }
            cfg->run_kind = *v;
            continue;
        }
        if (arg == "--campaign-kind") {
            if (!once(cfg->has_campaign_kind, "--campaign-kind")) return false;
            const auto v = next_value();
            if (!v || (*v != "none" && *v != "pilot" && *v != "final")) {
                *err = "--campaign-kind must be 'none', 'pilot' or 'final'";
                return false;
            }
            cfg->campaign_kind = *v;
            continue;
        }
        if (arg == "--iterations") {
            if (!once(cfg->has_iterations, "--iterations")) return false;
            const auto v = next_value();
            if (!v || !parse_int_arg(*v, &cfg->iterations) || cfg->iterations < 1) {
                *err = "--iterations must be an integer >= 1";
                return false;
            }
            continue;
        }
        if (arg == "--warmup-iterations") {
            if (!once(cfg->has_warmup_iterations, "--warmup-iterations")) return false;
            const auto v = next_value();
            if (!v || !parse_int_arg(*v, &cfg->warmup_iterations) || cfg->warmup_iterations < 0) {
                *err = "--warmup-iterations must be an integer >= 0";
                return false;
            }
            continue;
        }
        if (arg == "--repetitions") {
            if (!once(cfg->has_repetitions, "--repetitions")) return false;
            const auto v = next_value();
            if (!v || !parse_int_arg(*v, &cfg->repetitions) || cfg->repetitions < 1) {
                *err = "--repetitions must be an integer >= 1";
                return false;
            }
            continue;
        }
        *err = "unknown argument: " + arg;
        return false;
    }

    const bool any_other = cfg->has_run_kind || cfg->has_campaign_kind || cfg->has_iterations ||
                           cfg->has_warmup_iterations || cfg->has_repetitions;
    if (cfg->help) {
        if (any_other) { *err = "--help cannot be combined with other options"; return false; }
        return true;
    }
    if (!cfg->has_run_kind) { *err = "--run-kind is required"; return false; }
    if (!cfg->has_iterations) { *err = "--iterations is required"; return false; }
    if (!cfg->has_warmup_iterations) {
        *err = "--warmup-iterations is required";
        return false;
    }
    if (!cfg->has_repetitions) { *err = "--repetitions is required"; return false; }
    return true;
}

void print_csv_header() {
    std::printf(
        "schema_version,timestamp_utc,campaign_kind,run_kind,sample_index,"
        "execution_order,method,scale,cta_group,m,n,k,depth,iterations,warmup_iterations,"
        "repetitions,work_unit_count,umma_per_work_unit_per_iteration,total_umma_count,"
        "flops_per_umma,total_flops,kernel_time_ms,total_tflops,tflops_per_planned_active_sm,"
        "tflops_per_evidenced_active_sm,threads_per_cta,grid_blocks,cluster_size,cluster_count,"
        "hardware_sm_count,planned_active_sm_count,observed_unique_sm_count,"
        "occupancy_blocks_per_sm,max_active_clusters,shared_memory_reservation_bytes,"
        "unused_sm_count,coverage_status,residency_evidence,diagnostic_clock64_cycles_min,"
        "diagnostic_clock64_cycles_max,operand_path,input_type,accumulator_type,correctness,"
        "mismatches,max_abs_error\n");
}

struct RowContext {
    std::string timestamp_utc;
    std::string campaign_kind;
    std::string run_kind;
    int64_t sample_index = 0;
    int64_t execution_order = 0;
    int64_t iterations = 0;
    int64_t warmup_iterations = 0;
    int64_t repetitions = 0;
    int hardware_sm_count = 0;
    int64_t shared_memory_reservation_bytes = 0;
};

void print_csv_row(const Plan& plan, const LaunchResult& measurement, const RowContext& ctx) {
    const double seconds = static_cast<double>(measurement.kernel_time_ms) * 1e-3;
    const double total_tflops =
        seconds > 0.0 ? static_cast<double>(plan.total_flops) / seconds / 1e12 : 0.0;
    const double per_planned_sm =
        plan.planned_active_sm > 0 ? total_tflops / static_cast<double>(plan.planned_active_sm) : 0.0;
    const bool evidenced = coverage_is_evidenced(plan, measurement.observed_unique_sm);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << kSchemaVersion << ',' << ctx.timestamp_utc << ',' << ctx.campaign_kind << ','
        << ctx.run_kind << ',' << ctx.sample_index << ',' << ctx.execution_order
        << ',' << plan.method << ',' << plan.scale << ',' << plan.cta_group << ',' << plan.m << ','
        << kN << ',' << kK << ',' << kDepth << ',' << ctx.iterations << ','
        << ctx.warmup_iterations << ',' << ctx.repetitions << ',' << plan.work_units << ','
        << kDepth << ',' << plan.total_umma << ',' << plan.flops_per_umma << ','
        << plan.total_flops << ',' << measurement.kernel_time_ms << ',' << total_tflops << ','
        << per_planned_sm << ',';
    if (evidenced && measurement.observed_unique_sm > 0) {
        oss << (total_tflops / static_cast<double>(measurement.observed_unique_sm));
    } else {
        oss << kNotApplicable;
    }
    oss << ',' << kThreadsPerCta << ',' << plan.grid_blocks << ',' << plan.cluster_size << ',';
    if (plan.has_clusters) oss << plan.work_units;
    else oss << kNotApplicable;
    oss << ',' << ctx.hardware_sm_count << ',' << plan.planned_active_sm << ','
        << measurement.observed_unique_sm << ',' << plan.occupancy_blocks_per_sm << ',';
    if (plan.max_active_clusters >= 0) oss << plan.max_active_clusters;
    else oss << kNotApplicable;
    oss << ',' << ctx.shared_memory_reservation_bytes << ',' << plan.unused_sm << ','
        << coverage_status_of(plan, measurement.observed_unique_sm, ctx.hardware_sm_count) << ','
        << (plan.residency_proven ? "all_blocks_simultaneously_resident" : "not_established") << ','
        << measurement.cycles_min << ',' << measurement.cycles_max << ',' << kOperandPath << ','
        << kInputType << ',' << kAccumulatorType << ',' << "OK" << ','
        << measurement.validation.mismatches << ',' << measurement.validation.max_abs_error << '\n';
    std::fputs(oss.str().c_str(), stdout);
}

void allocate_plan(Plan& plan) {
    const int64_t d_elements =
        checked_mul_i64(plan.work_units, checked_mul_i64(plan.m, kN, "m*N"), "work_units*m*N");
    CUDA_CHECK_FATAL(cudaMalloc(&plan.d_out, checked_alloc_bytes(d_elements, sizeof(float), "d_out")));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.cycles,
                                checked_alloc_bytes(plan.work_units, sizeof(unsigned long long),
                                                    "cycles")));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.launch_ok,
                                checked_alloc_bytes(plan.grid_blocks, sizeof(int), "launch_ok")));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.smid,
                                checked_alloc_bytes(plan.grid_blocks, sizeof(int), "smid")));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.resident_ok,
                                checked_alloc_bytes(plan.grid_blocks, sizeof(int), "resident_ok")));
    CUDA_CHECK_FATAL(cudaMalloc(&plan.arrival, sizeof(unsigned int)));
    plan.host_d.assign(static_cast<size_t>(d_elements), 0.0f);
}

void free_plan(Plan& plan) {
    CUDA_CHECK_FATAL(cudaFree(plan.d_out));
    CUDA_CHECK_FATAL(cudaFree(plan.cycles));
    CUDA_CHECK_FATAL(cudaFree(plan.launch_ok));
    CUDA_CHECK_FATAL(cudaFree(plan.smid));
    CUDA_CHECK_FATAL(cudaFree(plan.resident_ok));
    CUDA_CHECK_FATAL(cudaFree(plan.arrival));
}

LaunchResult run_launch(Plan& plan, int64_t iterations, RunMode mode, size_t smem_bytes,
                        cudaEvent_t start, cudaEvent_t stop, bool measure, bool validate,
                        const std::vector<double>& reference) {
    LaunchResult result;
    const size_t block_ints = checked_alloc_bytes(plan.grid_blocks, sizeof(int), "block ints");
    CUDA_CHECK_FATAL(cudaMemset(plan.launch_ok, 0, block_ints));
    CUDA_CHECK_FATAL(cudaMemset(plan.smid, 0xFF, block_ints));
    CUDA_CHECK_FATAL(cudaMemset(plan.resident_ok, 0, block_ints));
    CUDA_CHECK_FATAL(cudaMemset(plan.arrival, 0, sizeof(unsigned int)));
    CUDA_CHECK_FATAL(cudaMemset(plan.cycles, 0,
                                checked_alloc_bytes(plan.work_units, sizeof(unsigned long long),
                                                    "cycles")));
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());

    if (measure) CUDA_CHECK_FATAL(cudaEventRecord(start));
    plan.kernel<<<plan.grid_blocks, kThreadsPerCta, smem_bytes>>>(
        iterations, mode, plan.d_out, plan.cycles, plan.launch_ok, plan.arrival, plan.resident_ok,
        plan.smid);
    CUDA_CHECK_FATAL(cudaGetLastError());
    if (measure) CUDA_CHECK_FATAL(cudaEventRecord(stop));
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    if (measure) {
        CUDA_CHECK_FATAL(cudaEventSynchronize(stop));
        CUDA_CHECK_FATAL(cudaEventElapsedTime(&result.kernel_time_ms, start, stop));
    }

    std::vector<int> launch_ok(static_cast<size_t>(plan.grid_blocks), 0);
    CUDA_CHECK_FATAL(cudaMemcpy(launch_ok.data(), plan.launch_ok, block_ints, cudaMemcpyDeviceToHost));
    for (int block = 0; block < plan.grid_blocks; ++block) {
        if (launch_ok[static_cast<size_t>(block)] != 1) {
            fail("launch-contract violation for %s/%s: block %d did not confirm grid=(%d,1,1) "
                 "cluster=(%d,1,1) block=(128,1,1)",
                 plan.method.c_str(), plan.scale.c_str(), block, plan.grid_blocks,
                 plan.cluster_size);
        }
    }

    std::vector<int> smid(static_cast<size_t>(plan.grid_blocks), -1);
    CUDA_CHECK_FATAL(cudaMemcpy(smid.data(), plan.smid, block_ints, cudaMemcpyDeviceToHost));
    result.observed_unique_sm = unique_count(smid);

    if (mode == RunMode::kResidency) {
        std::vector<int> resident(static_cast<size_t>(plan.grid_blocks), 0);
        CUDA_CHECK_FATAL(cudaMemcpy(resident.data(), plan.resident_ok, block_ints,
                                    cudaMemcpyDeviceToHost));
        plan.residency_ok_blocks = 0;
        for (int value : resident) plan.residency_ok_blocks += (value == 1) ? 1 : 0;
        plan.residency_proven = (plan.residency_ok_blocks == plan.grid_blocks);
        std::fprintf(stderr,
                     "umma_device_scaling: %s/%s residency_probe blocks=%d observed_all=%d "
                     "unique_sm=%d proven=%s sm_ids=[%s]\n",
                     plan.method.c_str(), plan.scale.c_str(), plan.grid_blocks,
                     plan.residency_ok_blocks, result.observed_unique_sm,
                     plan.residency_proven ? "true" : "false", join_ids(smid, 16).c_str());
        return result;
    }

    std::vector<unsigned long long> cycles(static_cast<size_t>(plan.work_units), 0ULL);
    CUDA_CHECK_FATAL(cudaMemcpy(cycles.data(), plan.cycles,
                                checked_alloc_bytes(plan.work_units, sizeof(unsigned long long),
                                                    "cycles"),
                                cudaMemcpyDeviceToHost));
    result.cycles_min = *std::min_element(cycles.begin(), cycles.end());
    result.cycles_max = *std::max_element(cycles.begin(), cycles.end());

    if (validate) {
        CUDA_CHECK_FATAL(cudaMemcpy(plan.host_d.data(), plan.d_out,
                                    plan.host_d.size() * sizeof(float), cudaMemcpyDeviceToHost));
        result.validation = validate_d(plan.host_d, plan.work_units, plan.m, reference);
    } else {
        result.validation.ok = true;
    }
    return result;
}

[[noreturn]] void die_on_validation_failure(const Plan& plan, const ValidationResult& validation) {
    std::fprintf(stderr,
                 "umma_device_scaling: ERROR: correctness validation FAILED for %s/%s: "
                 "mismatches=%lld first_index=%lld expected=%.1f obtained=%.1f max_abs_error=%.6g\n",
                 plan.method.c_str(), plan.scale.c_str(), (long long)validation.mismatches,
                 (long long)validation.first_mismatch_index, validation.first_mismatch_expected,
                 validation.first_mismatch_obtained, validation.max_abs_error);
    fail("correctness validation failed for %s/%s; no timing or CSV output was produced",
         plan.method.c_str(), plan.scale.c_str());
}

size_t choose_shared_memory_reservation(const GpuInfo& gpu) {
    const size_t operand_bytes_1sm = static_cast<size_t>(kMLocal) * kK * 2 + static_cast<size_t>(kN) * kK * 2;
    const size_t operand_bytes_2sm =
        static_cast<size_t>(kMLocal) * kK * 2 + static_cast<size_t>(kN / kClusterCtas) * kK * 2;
    const size_t operand_max = std::max(operand_bytes_1sm, operand_bytes_2sm);
    size_t reservation = gpu.shared_mem_per_multiprocessor / 2 + 1024;
    reservation = ((reservation + 127) / 128) * 128;
    if (gpu.shared_mem_per_block_optin > 0 &&
        reservation > static_cast<size_t>(gpu.shared_mem_per_block_optin)) {
        reservation = static_cast<size_t>(gpu.shared_mem_per_block_optin);
    }
    if (reservation < operand_max) {
        fail("shared-memory reservation %zu B is smaller than the %zu B of operands the kernel "
             "needs (sharedMemPerMultiprocessor=%zu, sharedMemPerBlockOptin=%d)",
             reservation, operand_max, gpu.shared_mem_per_multiprocessor,
             gpu.shared_mem_per_block_optin);
    }
    return reservation;
}

int occupancy_blocks_per_sm(KernelFn kernel, size_t reservation) {
    int blocks = 0;
    CUDA_CHECK_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, reinterpret_cast<const void*>(kernel), kThreadsPerCta, reservation));
    return blocks;
}

int max_active_clusters_for(KernelFn kernel, size_t reservation, int grid_blocks) {
    cudaLaunchConfig_t config{};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = kClusterCtas;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.gridDim = dim3(static_cast<unsigned int>(grid_blocks), 1, 1);
    config.blockDim = dim3(kThreadsPerCta, 1, 1);
    config.dynamicSmemBytes = reservation;
    config.stream = nullptr;
    config.attrs = attribute;
    config.numAttrs = 1;
    int clusters = 0;
    CUDA_CHECK_FATAL(cudaOccupancyMaxActiveClusters(&clusters,
                                                    reinterpret_cast<const void*>(kernel), &config));
    return clusters;
}

Plan make_plan(const char* method, const char* scale, int cta_group, int m, int cluster_size,
               int grid_blocks, bool device_scale, KernelFn kernel, const GpuInfo& gpu,
               int blocks_per_sm, int max_clusters, int64_t iterations) {
    Plan plan;
    plan.method = method;
    plan.scale = scale;
    plan.cta_group = cta_group;
    plan.m = m;
    plan.cluster_size = cluster_size;
    plan.has_clusters = (cluster_size > 1);
    plan.grid_blocks = grid_blocks;
    plan.device_scale = device_scale;
    plan.kernel = kernel;
    plan.occupancy_blocks_per_sm = blocks_per_sm;
    plan.max_active_clusters = plan.has_clusters ? max_clusters : -1;
    if (grid_blocks < 1 || grid_blocks % cluster_size != 0) {
        fail("invalid launch geometry for %s/%s: grid_blocks=%d is not a positive multiple of the "
             "cluster size %d",
             method, scale, grid_blocks, cluster_size);
    }
    if (grid_blocks > gpu.multi_processor_count) {
        fail("invalid launch geometry for %s/%s: grid_blocks=%d exceeds the device's %d SMs while "
             "occupancy is one CTA per SM",
             method, scale, grid_blocks, gpu.multi_processor_count);
    }
    plan.work_units = grid_blocks / cluster_size;
    plan.planned_active_sm = grid_blocks;
    plan.unused_sm = gpu.multi_processor_count - plan.planned_active_sm;
    plan.flops_per_umma =
        checked_mul_i64(checked_mul_i64(2, m, "2*M"), checked_mul_i64(kN, kK, "N*K"), "flops_per_umma");
    plan.total_umma = checked_mul_i64(checked_mul_i64(kDepth, iterations, "depth*iterations"),
                                      plan.work_units, "depth*iterations*work_units");
    plan.total_flops = checked_mul_i64(plan.flops_per_umma, plan.total_umma, "total_flops");
    return plan;
}

enum PlanIndex { kOneIsolated = 0, kTwoIsolated = 1, kOneDevice = 2, kTwoDevice = 3, kPlanCount = 4 };

std::vector<Plan> build_plans(const GpuInfo& gpu, size_t reservation, int64_t iterations) {
    if (gpu.multi_processor_count < kClusterCtas) {
        fail("the 2-SM arm needs at least %d SMs, device reports %d", kClusterCtas,
             gpu.multi_processor_count);
    }
    const int blocks_1sm = occupancy_blocks_per_sm(&umma_1sm_scaling_m128n256k16_d256, reservation);
    const int blocks_2sm = occupancy_blocks_per_sm(&umma_2sm_scaling_m256n256k16_d256, reservation);
    if (blocks_1sm != 1 || blocks_2sm != 1) {
        fail("the %zu B shared-memory reservation did not restrict occupancy to one CTA per SM "
             "(1-SM arm: %d blocks/SM, 2-SM arm: %d blocks/SM); one resident CTA per SM is required "
             "so that no SM hosts two 256-column Tensor Memory allocations",
             reservation, blocks_1sm, blocks_2sm);
    }

    const int theoretical_clusters = gpu.multi_processor_count / kClusterCtas;
    const int max_clusters = max_active_clusters_for(&umma_2sm_scaling_m256n256k16_d256, reservation,
                                                     theoretical_clusters * kClusterCtas);
    if (max_clusters < 1) {
        fail("cudaOccupancyMaxActiveClusters reported %d resident clusters", max_clusters);
    }
    const int device_clusters = std::min(theoretical_clusters, max_clusters);
    const int max_clusters_isolated =
        max_active_clusters_for(&umma_2sm_scaling_m256n256k16_d256, reservation, kClusterCtas);

    std::vector<Plan> plans;
    plans.push_back(make_plan("umma_1sm", "isolated", 1, kM1sm, 1, 1, false,
                              &umma_1sm_scaling_m128n256k16_d256, gpu, blocks_1sm, -1, iterations));
    plans.push_back(make_plan("umma_2sm", "isolated", 2, kM2sm, kClusterCtas, kClusterCtas, false,
                              &umma_2sm_scaling_m256n256k16_d256, gpu, blocks_2sm,
                              max_clusters_isolated, iterations));
    plans.push_back(make_plan("umma_1sm", "device_scale", 1, kM1sm, 1, gpu.multi_processor_count,
                              true, &umma_1sm_scaling_m128n256k16_d256, gpu, blocks_1sm, -1,
                              iterations));
    plans.push_back(make_plan("umma_2sm", "device_scale", 2, kM2sm, kClusterCtas,
                              device_clusters * kClusterCtas, true,
                              &umma_2sm_scaling_m256n256k16_d256, gpu, blocks_2sm, max_clusters,
                              iterations));

    const Plan& one = plans[kOneDevice];
    const Plan& two = plans[kTwoDevice];
    if (checked_mul_i64(one.flops_per_umma, 2, "2*flops_per_umma_1sm") != two.flops_per_umma) {
        fail("internal error: a 2-SM work unit must issue exactly twice a 1-SM work unit's FLOPs "
             "(%lld vs %lld)",
             (long long)one.flops_per_umma, (long long)two.flops_per_umma);
    }
    if (one.planned_active_sm == two.planned_active_sm && one.planned_active_sm % 2 == 0) {
        if (one.total_flops != two.total_flops) {
            fail("internal error: equal active-SM coverage (%d SMs) must give equal total FLOPs "
                 "(1-SM %lld vs 2-SM %lld)",
                 one.planned_active_sm, (long long)one.total_flops, (long long)two.total_flops);
        }
    }

    std::fprintf(stderr,
                 "umma_device_scaling: hardware_sm=%d reservation_bytes=%zu blocks_per_sm=%d "
                 "theoretical_clusters=%d max_active_clusters=%d device_clusters=%d "
                 "device_1sm_blocks=%d device_2sm_blocks=%d unused_sm_1sm=%d unused_sm_2sm=%d\n",
                 gpu.multi_processor_count, reservation, blocks_1sm, theoretical_clusters,
                 max_clusters, device_clusters, one.grid_blocks, two.grid_blocks, one.unused_sm,
                 two.unused_sm);
    if (two.unused_sm > 0) {
        std::fprintf(stderr,
                     "umma_device_scaling: NOTE: the 2-SM device-scale configuration leaves %d of "
                     "%d SMs unused (odd SM count or a cluster-occupancy limit); its totals are "
                     "NOT directly comparable with the 1-SM configuration's %d active SMs\n",
                     two.unused_sm, gpu.multi_processor_count, one.planned_active_sm);
    }
    return plans;
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig cli;
    std::string parse_err;
    if (!parse_cli(argc, argv, &cli, &parse_err)) {
        std::fprintf(stderr, "umma_device_scaling: ERROR: %s\n", parse_err.c_str());
        print_usage(stderr);
        return 2;
    }
    if (cli.help) {
        print_usage(stdout);
        return 0;
    }

    const GpuInfo gpu = query_gpu_info();
    const std::vector<double> reference = build_reference();
    const size_t reservation = choose_shared_memory_reservation(gpu);
    CUDA_CHECK_FATAL(cudaFuncSetAttribute(reinterpret_cast<const void*>(&umma_1sm_scaling_m128n256k16_d256),
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(reservation)));
    CUDA_CHECK_FATAL(cudaFuncSetAttribute(reinterpret_cast<const void*>(&umma_2sm_scaling_m256n256k16_d256),
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(reservation)));

    std::fprintf(stderr,
                 "umma_device_scaling: run_kind=%s campaign_kind=%s iterations=%lld "
                 "warmup_iterations=%lld repetitions=%lld n=%d depth=%d\n",
                 cli.run_kind.c_str(), cli.campaign_kind.c_str(), (long long)cli.iterations,
                 (long long)cli.warmup_iterations, (long long)cli.repetitions, kN, kDepth);

    std::vector<Plan> plans = build_plans(gpu, reservation, cli.iterations);
    for (Plan& plan : plans) allocate_plan(plan);

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK_FATAL(cudaEventCreate(&start));
    CUDA_CHECK_FATAL(cudaEventCreate(&stop));

    // Establish simultaneous residency before correctness and timing.
    for (Plan& plan : plans) {
        run_launch(plan, 1, RunMode::kResidency, reservation, nullptr, nullptr, false, false,
                   reference);
        if (plan.device_scale && !plan.residency_proven) {
            std::fprintf(stderr,
                         "umma_device_scaling: WARNING: %s/%s could not establish simultaneous "
                         "residency for all %d blocks (%d confirmed); its rows are recorded as "
                         "incomplete_coverage\n",
                         plan.method.c_str(), plan.scale.c_str(), plan.grid_blocks,
                         plan.residency_ok_blocks);
        }
    }

    // Validate all work units before collecting any timed sample.
    for (Plan& plan : plans) {
        const LaunchResult result = run_launch(plan, cli.iterations, RunMode::kUntimed, reservation,
                                               nullptr, nullptr, false, true, reference);
        if (!result.validation.ok) die_on_validation_failure(plan, result.validation);
        std::fprintf(stderr,
                     "umma_device_scaling: %s/%s correctness=OK mismatches=0 work_units=%d "
                     "unique_sm=%d (pre-timing check)\n",
                     plan.method.c_str(), plan.scale.c_str(), plan.work_units,
                     result.observed_unique_sm);
    }

    for (Plan& plan : plans) {
        for (int64_t w = 0; w < cli.warmup_iterations; ++w) {
            run_launch(plan, cli.iterations, RunMode::kTimed, reservation, start, stop, true, false,
                       reference);
        }
    }

    RowContext ctx;
    ctx.campaign_kind = cli.campaign_kind;
    ctx.run_kind = cli.run_kind;
    ctx.iterations = cli.iterations;
    ctx.warmup_iterations = cli.warmup_iterations;
    ctx.repetitions = cli.repetitions;
    ctx.hardware_sm_count = gpu.multi_processor_count;
    ctx.shared_memory_reservation_bytes = static_cast<int64_t>(reservation);

    print_csv_header();
    int64_t execution_order = 0;
    for (int64_t rep = 0; rep < cli.repetitions; ++rep) {
        // Alternate launch order to reduce thermal and clock-order bias.
        const bool forward = (rep % 2 == 0);
        for (int step = 0; step < kPlanCount; ++step) {
            const int index = forward ? step : (kPlanCount - 1 - step);
            Plan& plan = plans[static_cast<size_t>(index)];
            const LaunchResult result = run_launch(plan, cli.iterations, RunMode::kTimed,
                                                   reservation, start, stop, true, true, reference);
            if (!result.validation.ok) die_on_validation_failure(plan, result.validation);
            if (!(result.kernel_time_ms > 0.0f)) {
                fail("internal error: CUDA-event kernel time was not greater than zero for %s/%s",
                     plan.method.c_str(), plan.scale.c_str());
            }
            ctx.timestamp_utc = now_utc_iso8601();
            ctx.sample_index = rep;
            ctx.execution_order = execution_order++;
            print_csv_row(plan, result, ctx);
        }
        std::fflush(stdout);
    }

    for (Plan& plan : plans) free_plan(plan);
    CUDA_CHECK_FATAL(cudaEventDestroy(start));
    CUDA_CHECK_FATAL(cudaEventDestroy(stop));
    return 0;
}
