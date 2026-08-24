// Two-SM BF16 UMMA throughput, timed with the leader's %clock64 counter.

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
constexpr int kClusterCtas = 2;       // exactly one static two-CTA cluster
constexpr int kGridBlocks = 2;        // grid == cluster: exactly one cluster
constexpr int kCtaGroup = 2;
constexpr int kMGlobal = 256;         // joint M across the CTA pair (idesc/FLOP accounting only)
constexpr int kMLocal = 128;          // each CTA's own local A/D row count (physical SMEM/TMEM shape)
constexpr int kK = 16;
constexpr const char* kSchemaVersion = "1";
constexpr const char* kMethodName = "umma_2sm";
constexpr const char* kOperandPath = "smem_smem";
constexpr const char* kInputType = "bf16";
constexpr const char* kAccumulatorType = "fp32";

enum class TimingMode : int32_t { kUntimed = 0, kTimed = 1 };

int g_cleanup_failures = 0;

[[noreturn]] void fail(const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "umma_2sm: ERROR: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::exit(1);
}

#define CUDA_CHECK_FATAL(call)                                                \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fail("cuda_error=%s detail=\"%s\" at %s:%d", cudaGetErrorName(err_), \
                 cudaGetErrorString(err_), __FILE__, __LINE__);               \
        }                                                                     \
    } while (0)

int64_t checked_mul_i64(int64_t a, int64_t b, const char* what) {
    const __int128 result = static_cast<__int128>(a) * static_cast<__int128>(b);
    if (result > static_cast<__int128>(INT64_MAX) || result < static_cast<__int128>(INT64_MIN)) {
        fail("integer overflow computing %s (a=%lld b=%lld)", what, (long long)a, (long long)b);
    }
    return static_cast<int64_t>(result);
}

__device__ __forceinline__ int smem_core_tile_index(int group_idx, int pos_in_group, int k) {
    constexpr int kSboElem = 128;  // 8 rows * 16 K-elements
    constexpr int kLboElem = 64;   // 8 rows * 8 (T) K-elements
    constexpr int kT = 8;
    const int chunk = k / kT;
    const int t = k % kT;
    return group_idx * kSboElem + chunk * kLboElem + pos_in_group * kT + t;
}

__device__ __forceinline__ uint64_t make_smem_descriptor(const void* smem_ptr) {
    constexpr uint32_t kLboBytes = 128;  // 64 elements * 2 bytes/element
    constexpr uint32_t kSboBytes = 256;  // 128 elements * 2 bytes/element
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFFu);
    desc |= static_cast<uint64_t>((kLboBytes >> 4) & 0x3FFFu) << 16;
    desc |= static_cast<uint64_t>((kSboBytes >> 4) & 0x3FFFu) << 32;
    desc |= (UINT64_C(1) << 46);
    return desc;
}

template <int N>
__device__ constexpr uint32_t make_instruction_descriptor() {
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128, or 256");
    static_assert(((static_cast<uint32_t>(N) >> 3) & ~0x3Fu) == 0, "N field overflows bits 17-22");
    static_assert(((static_cast<uint32_t>(kMGlobal) >> 4) & ~0x1Fu) == 0, "M field overflows bits 24-28");
    uint32_t desc = 0;
    desc |= (1u << 4);                                       // bits 4-5: dtype = FP32
    desc |= (1u << 7);                                        // bits 7-9: atype = BF16
    desc |= (1u << 10);                                       // bits 10-12: btype = BF16
    desc |= ((static_cast<uint32_t>(N) >> 3) << 17);           // bits 17-22: N >> 3
    desc |= ((static_cast<uint32_t>(kMGlobal) >> 4) << 24);    // bits 24-28: M >> 4 (M = 256)
    return desc;
}

template <int N>
__device__ constexpr bool validate_instruction_descriptor() {
    constexpr uint32_t desc = make_instruction_descriptor<N>();
    constexpr uint32_t sparsity_selector = desc & 0x3u;          // bits 0-1
    constexpr uint32_t sparsity = (desc >> 2) & 0x1u;             // bit 2
    constexpr uint32_t saturate = (desc >> 3) & 0x1u;             // bit 3
    constexpr uint32_t dtype = (desc >> 4) & 0x3u;                // bits 4-5
    constexpr uint32_t reserved6 = (desc >> 6) & 0x1u;            // bit 6
    constexpr uint32_t atype = (desc >> 7) & 0x7u;                // bits 7-9
    constexpr uint32_t btype = (desc >> 10) & 0x7u;               // bits 10-12
    constexpr uint32_t negate_a = (desc >> 13) & 0x1u;            // bit 13
    constexpr uint32_t negate_b = (desc >> 14) & 0x1u;            // bit 14
    constexpr uint32_t transpose_a = (desc >> 15) & 0x1u;         // bit 15
    constexpr uint32_t transpose_b = (desc >> 16) & 0x1u;         // bit 16
    constexpr uint32_t n_field = (desc >> 17) & 0x3Fu;            // bits 17-22
    constexpr uint32_t reserved23 = (desc >> 23) & 0x1u;          // bit 23
    constexpr uint32_t m_field = (desc >> 24) & 0x1Fu;            // bits 24-28
    constexpr uint32_t reserved29 = (desc >> 29) & 0x1u;          // bit 29
    constexpr uint32_t ws_shift = (desc >> 30) & 0x3u;            // bits 30-31
    return sparsity_selector == 0 && sparsity == 0 && saturate == 0 && dtype == 1 &&
           reserved6 == 0 && atype == 1 && btype == 1 && negate_a == 0 && negate_b == 0 &&
           transpose_a == 0 && transpose_b == 0 &&
           n_field == (static_cast<uint32_t>(N) >> 3) && reserved23 == 0 &&
           m_field == (static_cast<uint32_t>(kMGlobal) >> 4) && reserved29 == 0 && ws_shift == 0;
}

static_assert(validate_instruction_descriptor<64>(), "N=64 instruction descriptor is malformed");
static_assert(validate_instruction_descriptor<128>(), "N=128 instruction descriptor is malformed");
static_assert(validate_instruction_descriptor<256>(), "N=256 instruction descriptor is malformed");
static_assert(make_instruction_descriptor<64>() != 0, "M=256 field must be nonzero in the descriptor");


// One two-CTA cluster issues cta_group::2 UMMA instructions.
__device__ __forceinline__ void issue_one_umma_2sm(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                     uint32_t idesc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t"
        "}\n\t"
        :
        : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
        : "memory");
}

__device__ __forceinline__ void commit_umma_2sm_multicast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "h"(cta_mask)
        : "memory");
}

__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t dst_smem_addr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(dst_smem_addr), "r"(n_cols)
                 : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" : : "r"(taddr), "r"(n_cols) : "memory");
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

constexpr uint32_t kTmemLaneShift = 16;        // PTX ISA 9.3 9.7.17.1.1: lane index occupies bits 31-16.
constexpr uint32_t kTmemRowsPerWarp = 32;      // PTX ISA 9.3 9.7.17.8.1: each warp owns a 32-lane chunk.
constexpr uint32_t kTmemColsPerFragment = 32;  // This kernel's fixed 32-column tcgen05.ld.x32 fragment width.

__device__ __forceinline__ uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {
    const uint32_t lane_contribution = (static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift;
    const uint32_t column_contribution = static_cast<uint32_t>(frag) * kTmemColsPerFragment;
    return tmem_base + lane_contribution + column_contribution;
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

constexpr int kExpectedGridDim = kGridBlocks;
constexpr int kExpectedBlockDimX = kThreadsPerCta;
constexpr uint32_t kExpectedClusterCtas = kClusterCtas;

__device__ __forceinline__ bool launch_contract_is_valid(uint32_t cluster_nctarank, uint32_t cluster_ctarank) {
    return gridDim.x == kExpectedGridDim && gridDim.y == 1 && gridDim.z == 1 &&
           blockDim.x == kExpectedBlockDimX && blockDim.y == 1 && blockDim.z == 1 &&
           cluster_nctarank == kExpectedClusterCtas && cluster_ctarank < kExpectedClusterCtas;
}

template <int N, int DEPTH>
__device__ __forceinline__ void umma_2sm_body(int64_t iterations, TimingMode timing_mode,
                                               float* __restrict__ g_d_out,
                                               unsigned long long* __restrict__ g_elapsed_cycles,
                                               int* __restrict__ g_launch_ok) {
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128, or 256");
    static_assert(DEPTH == 4 || DEPTH == 16 || DEPTH == 64 || DEPTH == 256,
                  "DEPTH must be 4, 16, 64, or 256");

    const int tid = threadIdx.x;
    const uint32_t cluster_ctarank = cuda::ptx::get_sreg_cluster_ctarank();
    const uint32_t cluster_nctarank = cuda::ptx::get_sreg_cluster_nctarank();

    if (!launch_contract_is_valid(cluster_nctarank, cluster_ctarank)) {
        if (tid == 0) {
            if (cluster_ctarank == 0) g_launch_ok[0] = 0;
            else if (cluster_ctarank == 1) g_launch_ok[1] = 0;
        }
        return;
    }
    if (tid == 0) {
        if (cluster_ctarank == 0) g_launch_ok[0] = 1;
        else if (cluster_ctarank == 1) g_launch_ok[1] = 1;
    }

    const int cta_rank = static_cast<int>(cluster_ctarank);  // 0 or 1, confirmed valid by the guard above

    constexpr int kNLocal = N / kClusterCtas;
    static_assert(N % kClusterCtas == 0, "N must divide evenly across the two peer CTAs");
    constexpr int kABytes = kMLocal * kK * 2;  // BF16 = 2 bytes/element; 128 local A rows per CTA

    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);

    const int warp_id = tid / 32;

    for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) {
        const int local_row = idx / kK;
        const int k = idx % kK;
        const int global_row = cta_rank * kMLocal + local_row;
        const int value = ((global_row + 3 * k) % 7) - 3;
        A[smem_core_tile_index(local_row / 8, local_row % 8, k)] = __float2bfloat16(static_cast<float>(value));
    }
    for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) {
        const int local_col = idx / kK;
        const int k = idx % kK;
        const int global_col = cta_rank * kNLocal + local_col;
        const int value = ((2 * k + global_col) % 5) - 2;
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
    if (tid < 32) {
        is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    }

    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    if (warp_id == 0) {
        const uint32_t tmem_addr_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared));
        tcgen05_alloc_2sm(tmem_addr_smem, static_cast<uint32_t>(N));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);

    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_instruction_descriptor<N>();
    const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    unsigned long long elapsed_cycles = 0;
    uint64_t start_clock = 0, end_clock = 0;
    if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");
    }

    uint32_t parity = 0;
    for (int64_t it = 0; it < iterations; ++it) {
        if (is_leader) {
            if (cta_rank == 0) {
                issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/0);
#pragma unroll
                for (int m = 1; m < DEPTH; ++m) {
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

    if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
        elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
    }
    if (is_leader && cta_rank == 0) {
        g_elapsed_cycles[0] = elapsed_cycles;
    }

    __syncthreads();
    tcgen05_fence_after_thread_sync();

    constexpr int kFragments = N / 32;
    static_assert(kFragments * 32 == N, "N must be a multiple of 32 for 32-column fragments");
    const int lane = tid % 32;
    const int local_row = warp_id * 32 + lane;
    const int global_row = cta_rank * kMLocal + local_row;
#pragma unroll
    for (int frag = 0; frag < kFragments; ++frag) {
        uint32_t regs[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);
        tcgen05_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            g_d_out[static_cast<int64_t>(global_row) * N + frag * 32 + i] = __uint_as_float(regs[i]);
        }
    }

    __syncthreads();
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    if (warp_id == 0) {
        tcgen05_dealloc_2sm(tmem_d, static_cast<uint32_t>(N));
        tcgen05_relinquish_alloc_permit_2sm();
    }

    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     :
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar)))
                     : "memory");
    }
}

}  // namespace

#define UMMA_2SM_DEFINE_KERNEL(N, DEPTH)                                                            \
    extern "C" __global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128) void umma_2sm_m256n##N##k16_d##DEPTH( \
        int64_t iterations, TimingMode timing_mode, float* g_d_out,                                  \
        unsigned long long* g_elapsed_cycles, int* g_launch_ok) {                                    \
        umma_2sm_body<N, DEPTH>(iterations, timing_mode, g_d_out, g_elapsed_cycles, g_launch_ok);     \
    }

UMMA_2SM_DEFINE_KERNEL(64, 4)
UMMA_2SM_DEFINE_KERNEL(64, 16)
UMMA_2SM_DEFINE_KERNEL(64, 64)
UMMA_2SM_DEFINE_KERNEL(64, 256)
UMMA_2SM_DEFINE_KERNEL(128, 4)
UMMA_2SM_DEFINE_KERNEL(128, 16)
UMMA_2SM_DEFINE_KERNEL(128, 64)
UMMA_2SM_DEFINE_KERNEL(128, 256)
UMMA_2SM_DEFINE_KERNEL(256, 4)
UMMA_2SM_DEFINE_KERNEL(256, 16)
UMMA_2SM_DEFINE_KERNEL(256, 64)
UMMA_2SM_DEFINE_KERNEL(256, 256)

#undef UMMA_2SM_DEFINE_KERNEL

namespace {

typedef void (*KernelFn)(int64_t, TimingMode, float*, unsigned long long*, int*);

struct Specialization {
    int n = 0;
    int depth = 0;
    KernelFn kernel = nullptr;
    const char* symbol = "";
};

#define UMMA_2SM_SPEC_ENTRY(N, DEPTH) \
    { N, DEPTH, &umma_2sm_m256n##N##k16_d##DEPTH, "umma_2sm_m256n" #N "k16_d" #DEPTH }

constexpr Specialization kSpecializations[12] = {
    UMMA_2SM_SPEC_ENTRY(64, 4),   UMMA_2SM_SPEC_ENTRY(64, 16),   UMMA_2SM_SPEC_ENTRY(64, 64),
    UMMA_2SM_SPEC_ENTRY(64, 256), UMMA_2SM_SPEC_ENTRY(128, 4),   UMMA_2SM_SPEC_ENTRY(128, 16),
    UMMA_2SM_SPEC_ENTRY(128, 64), UMMA_2SM_SPEC_ENTRY(128, 256), UMMA_2SM_SPEC_ENTRY(256, 4),
    UMMA_2SM_SPEC_ENTRY(256, 16), UMMA_2SM_SPEC_ENTRY(256, 64),  UMMA_2SM_SPEC_ENTRY(256, 256),
};

#undef UMMA_2SM_SPEC_ENTRY

const Specialization* find_spec(int n, int depth) {
    for (const auto& s : kSpecializations) {
        if (s.n == n && s.depth == depth) return &s;
    }
    return nullptr;
}

int32_t reference_a(int global_row, int k) { return ((global_row + 3 * k) % 7) - 3; }
int32_t reference_b(int k, int col) { return ((2 * k + col) % 5) - 2; }

double reference_d(int global_row, int col, int depth) {
    int64_t sum = 0;
    for (int k = 0; k < kK; ++k) {
        sum += static_cast<int64_t>(reference_a(global_row, k)) * static_cast<int64_t>(reference_b(k, col));
    }
    return static_cast<double>(depth) * static_cast<double>(sum);
}

struct ValidationResult {
    bool ok = false;
    int64_t mismatches = 0;
    int64_t first_mismatch_index = -1;
    double first_mismatch_expected = 0.0;
    double first_mismatch_obtained = 0.0;
    double max_abs_error = 0.0;
};

ValidationResult validate_d(const std::vector<float>& d_out, int n, int depth) {
    ValidationResult r;
    for (int global_row = 0; global_row < kMGlobal; ++global_row) {
        for (int col = 0; col < n; ++col) {
            const double expected = reference_d(global_row, col, depth);
            const double obtained =
                static_cast<double>(d_out[static_cast<size_t>(global_row) * n + col]);
            const double abs_error = std::fabs(obtained - expected);
            if (abs_error > r.max_abs_error) r.max_abs_error = abs_error;
            if (obtained != expected) {
                if (r.mismatches == 0) {
                    r.first_mismatch_index = static_cast<int64_t>(global_row) * n + col;
                    r.first_mismatch_expected = expected;
                    r.first_mismatch_obtained = obtained;
                }
                ++r.mismatches;
            }
        }
    }
    r.ok = (r.mismatches == 0);
    return r;
}

struct CliConfig {
    bool help = false;
    bool has_run_kind = false;
    std::string run_kind;
    bool has_n = false;
    int n = 0;
    bool has_depth = false;
    int depth = 0;
    bool has_iterations = false;
    int64_t iterations = 0;
    bool has_warmup_iterations = false;
    int64_t warmup_iterations = 0;
    bool has_repetitions = false;
    int64_t repetitions = 0;
};

void print_usage(std::FILE* out) {
    std::fprintf(out,
        "Usage: umma_2sm --run-kind {smoke,benchmark} --n {64,128,256}\n"
        "       --depth {4,16,64,256} --iterations N --warmup-iterations N\n"
        "       --repetitions N\n");
}

bool parse_cli(int argc, char** argv, CliConfig* cfg, std::string* err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](void) -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };

        if (arg == "--help") {
            if (cfg->help) { *err = "--help specified more than once"; return false; }
            cfg->help = true;
            continue;
        }
        if (arg == "--run-kind") {
            if (cfg->has_run_kind) { *err = "--run-kind specified more than once"; return false; }
            const auto v = next_value();
            if (!v || (*v != "smoke" && *v != "benchmark")) {
                *err = "--run-kind must be 'smoke' or 'benchmark'";
                return false;
            }
            cfg->run_kind = *v;
            cfg->has_run_kind = true;
            continue;
        }
        if (arg == "--n") {
            if (cfg->has_n) { *err = "--n specified more than once"; return false; }
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || (iv != 64 && iv != 128 && iv != 256)) {
                *err = "--n must be one of 64, 128, 256";
                return false;
            }
            cfg->n = static_cast<int>(iv);
            cfg->has_n = true;
            continue;
        }
        if (arg == "--depth") {
            if (cfg->has_depth) { *err = "--depth specified more than once"; return false; }
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || (iv != 4 && iv != 16 && iv != 64 && iv != 256)) {
                *err = "--depth must be one of 4, 16, 64, 256";
                return false;
            }
            cfg->depth = static_cast<int>(iv);
            cfg->has_depth = true;
            continue;
        }
        if (arg == "--iterations") {
            if (cfg->has_iterations) { *err = "--iterations specified more than once"; return false; }
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 1) {
                *err = "--iterations must be an integer >= 1";
                return false;
            }
            cfg->iterations = iv;
            cfg->has_iterations = true;
            continue;
        }
        if (arg == "--warmup-iterations") {
            if (cfg->has_warmup_iterations) { *err = "--warmup-iterations specified more than once"; return false; }
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 0) {
                *err = "--warmup-iterations must be an integer >= 0";
                return false;
            }
            cfg->warmup_iterations = iv;
            cfg->has_warmup_iterations = true;
            continue;
        }
        if (arg == "--repetitions") {
            if (cfg->has_repetitions) { *err = "--repetitions specified more than once"; return false; }
            const auto v = next_value();
            int64_t iv = 0;
            if (!v || !parse_int_arg(*v, &iv) || iv < 1) {
                *err = "--repetitions must be an integer >= 1";
                return false;
            }
            cfg->repetitions = iv;
            cfg->has_repetitions = true;
            continue;
        }
        *err = "unknown argument: " + arg;
        return false;
    }

    if (cfg->help) {
        if (cfg->has_run_kind || cfg->has_n || cfg->has_depth || cfg->has_iterations ||
            cfg->has_warmup_iterations || cfg->has_repetitions) {
            *err = "--help cannot be combined with other options";
            return false;
        }
        return true;
    }
    if (!cfg->has_run_kind) { *err = "--run-kind is required"; return false; }
    if (!cfg->has_n) { *err = "--n is required"; return false; }
    if (!cfg->has_depth) { *err = "--depth is required"; return false; }
    if (!cfg->has_iterations) { *err = "--iterations is required"; return false; }
    if (!cfg->has_warmup_iterations) {
        *err = "--warmup-iterations is required";
        return false;
    }
    if (!cfg->has_repetitions) { *err = "--repetitions is required"; return false; }
    return true;
}

struct CsvRow {
    std::string timestamp_utc;
    std::string run_kind;
    int64_t sample_index = 0;
    int n = 0;
    int depth = 0;
    int64_t iterations = 0;
    int64_t warmup_iterations = 0;
    int64_t repetitions = 0;
    int64_t total_umma = 0;
    int64_t flops_per_umma = 0;
    int64_t total_flops = 0;
    unsigned long long elapsed_cycles = 0;
};

void print_csv_header() {
    std::printf(
        "schema_version,timestamp_utc,run_kind,method,sample_index,cta_group,m,n,k,"
        "depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,"
        "flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,"
        "grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,"
        "max_abs_error\n");
}

void print_csv_row(const CsvRow& r) {
    const double cycles_per_umma = r.total_umma > 0
        ? static_cast<double>(r.elapsed_cycles) / static_cast<double>(r.total_umma)
        : 0.0;
    const double flops_per_cycle = r.elapsed_cycles > 0
        ? static_cast<double>(r.total_flops) / static_cast<double>(r.elapsed_cycles)
        : 0.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << kSchemaVersion << ',' << r.timestamp_utc << ',' << r.run_kind << ','
        << kMethodName << ',' << r.sample_index << ',' << kCtaGroup << ',' << kMGlobal << ',' << r.n << ','
        << kK << ',' << r.depth << ',' << r.iterations << ',' << r.warmup_iterations << ','
        << r.repetitions << ',' << r.depth << ',' << r.total_umma << ',' << r.flops_per_umma << ','
        << r.total_flops << ',' << r.elapsed_cycles << ',' << cycles_per_umma << ','
        << flops_per_cycle << ',' << kThreadsPerCta << ',' << kGridBlocks << ',' << r.n << ','
        << kOperandPath << ',' << kInputType << ',' << kAccumulatorType << ',' << "OK" << ',' << 0
        << ',' << 0 << '\n';
    std::fputs(oss.str().c_str(), stdout);
}

struct RunResult {
    ValidationResult validation;
    unsigned long long elapsed_cycles = 0;
};

RunResult run_once(const Specialization& spec, int64_t iterations, TimingMode mode) {
    RunResult result;
    const size_t d_elems = static_cast<size_t>(kMGlobal) * static_cast<size_t>(spec.n);
    float* d_out_device = nullptr;
    unsigned long long* cycles_device = nullptr;
    int* launch_ok_device = nullptr;
    CUDA_CHECK_FATAL(cudaMalloc(&d_out_device, d_elems * sizeof(float)));
    CUDA_CHECK_FATAL(cudaMalloc(&cycles_device, sizeof(unsigned long long)));
    CUDA_CHECK_FATAL(cudaMalloc(&launch_ok_device, kClusterCtas * sizeof(int)));
    CUDA_CHECK_FATAL(cudaMemset(cycles_device, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FATAL(cudaMemset(launch_ok_device, 0, kClusterCtas * sizeof(int)));

    const int n_local = spec.n / kClusterCtas;
    const int smem_bytes = kMLocal * kK * 2 + n_local * kK * 2;
    spec.kernel<<<kGridBlocks, kThreadsPerCta, static_cast<size_t>(smem_bytes)>>>(
        iterations, mode, d_out_device, cycles_device, launch_ok_device);
    CUDA_CHECK_FATAL(cudaGetLastError());
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());

    int launch_ok_host[kClusterCtas] = {0, 0};
    CUDA_CHECK_FATAL(cudaMemcpy(launch_ok_host, launch_ok_device, sizeof(launch_ok_host),
                                 cudaMemcpyDeviceToHost));
    std::vector<float> d_out_host(d_elems);
    CUDA_CHECK_FATAL(
        cudaMemcpy(d_out_host.data(), d_out_device, d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FATAL(cudaMemcpy(&result.elapsed_cycles, cycles_device, sizeof(result.elapsed_cycles),
                                 cudaMemcpyDeviceToHost));
    CUDA_CHECK_FATAL(cudaFree(d_out_device));
    CUDA_CHECK_FATAL(cudaFree(cycles_device));
    CUDA_CHECK_FATAL(cudaFree(launch_ok_device));

    if (launch_ok_host[0] != 1 || launch_ok_host[1] != 1) {
        fail("launch-contract violation for %s: kernel did not confirm grid=(2,1,1) "
             "cluster=(2,1,1) block=(128,1,1) (launch_ok=[%d,%d])",
             spec.symbol, launch_ok_host[0], launch_ok_host[1]);
    }

    result.validation = validate_d(d_out_host, spec.n, spec.depth);
    return result;
}

[[noreturn]] void die_on_validation_failure(const Specialization& spec, const ValidationResult& validation) {
    std::fprintf(stderr,
                 "umma_2sm: ERROR: correctness validation FAILED for %s: mismatches=%lld "
                 "first_index=%lld expected=%.1f obtained=%.1f max_abs_error=%.6g\n",
                 spec.symbol, (long long)validation.mismatches, (long long)validation.first_mismatch_index,
                 validation.first_mismatch_expected, validation.first_mismatch_obtained,
                 validation.max_abs_error);
    fail("correctness validation failed for %s; no timing or CSV output was produced", spec.symbol);
}

void run_untimed_or_die(const Specialization& spec, int64_t iterations) {
    const RunResult result = run_once(spec, iterations, TimingMode::kUntimed);
    if (!result.validation.ok) die_on_validation_failure(spec, result.validation);
}

unsigned long long run_timed_or_die(const Specialization& spec, int64_t iterations) {
    const RunResult result = run_once(spec, iterations, TimingMode::kTimed);
    if (!result.validation.ok) die_on_validation_failure(spec, result.validation);
    if (result.elapsed_cycles == 0) {
        fail("internal error: elapsed_cycles was not greater than zero for %s", spec.symbol);
    }
    return result.elapsed_cycles;
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig cli;
    std::string parse_err;
    if (!parse_cli(argc, argv, &cli, &parse_err)) {
        std::fprintf(stderr, "umma_2sm: ERROR: %s\n", parse_err.c_str());
        print_usage(stderr);
        return 2;
    }
    if (cli.help) {
        print_usage(stdout);
        return 0;
    }

    benchmark_device_properties();

    const Specialization* spec = find_spec(cli.n, cli.depth);
    if (spec == nullptr) {
        fail("internal error: no specialization for n=%d depth=%d", cli.n, cli.depth);
    }

    std::fprintf(stderr,
                 "umma_2sm: run_kind=%s n=%d depth=%d iterations=%lld warmup_iterations=%lld "
                 "repetitions=%lld\n",
                 cli.run_kind.c_str(), cli.n, cli.depth, (long long)cli.iterations,
                 (long long)cli.warmup_iterations, (long long)cli.repetitions);

    // Reject incorrect output before warm-up or timed samples.
    run_untimed_or_die(*spec, cli.iterations);
    std::fprintf(stderr, "umma_2sm: correctness=OK mismatches=0 (pre-timing check)\n");

    for (int64_t w = 0; w < cli.warmup_iterations; ++w) {
        run_untimed_or_die(*spec, cli.iterations);
    }

    print_csv_header();
    const int64_t flops_per_umma =
        checked_mul_i64(checked_mul_i64(2, kMGlobal, "2*M"), checked_mul_i64(spec->n, kK, "N*K"), "flops_per_umma");
    const int64_t total_umma = checked_mul_i64(spec->depth, cli.iterations, "depth*iterations");
    const int64_t total_flops = checked_mul_i64(flops_per_umma, total_umma, "flops_per_umma*total_umma");

    for (int64_t rep = 0; rep < cli.repetitions; ++rep) {
        const unsigned long long elapsed_cycles = run_timed_or_die(*spec, cli.iterations);

        CsvRow row;
        row.timestamp_utc = now_utc_iso8601();
        row.run_kind = cli.run_kind;
        row.sample_index = rep;
        row.n = spec->n;
        row.depth = spec->depth;
        row.iterations = cli.iterations;
        row.warmup_iterations = cli.warmup_iterations;
        row.repetitions = cli.repetitions;
        row.total_umma = total_umma;
        row.flops_per_umma = flops_per_umma;
        row.total_flops = total_flops;
        row.elapsed_cycles = elapsed_cycles;
        print_csv_row(row);
    }

    if (g_cleanup_failures != 0) {
        std::fprintf(stderr, "umma_2sm: ERROR: %d resource cleanup failure(s) occurred\n",
                     g_cleanup_failures);
        return 1;
    }
    return 0;
}
