// 1-SM BF16 UMMA instruction microbenchmark: the single-CTA, cta_group::1
// arm of the "BF16 UMMA throughput" experiment.
//
// Frozen experimental contract: tcgen05.mma, kind::f16, cta_group::1,
// BF16 x BF16 -> FP32, M=128, K=16 (implied by kind::f16 dense BF16), N in
// {64,128,256}, depth in {4,16,64,256}, A and B in SMEM, D in TMEM, exactly
// one CTA of 128 threads, exactly twelve specializations.
//
// Primary normative source: NVIDIA PTX ISA 9.3
// (https://docs.nvidia.com/cuda/parallel-thread-execution/), chapter
// "9.7.17. TensorCore 5th Generation Family Instructions". Secondary
// conceptual reference (adapted, not copied, and independently checked
// against the PTX ISA): the pinned revision of
// SemiAnalysisAI/microbench-blackwell/umma_throughput/umma_tput.cu. This
// file's shared-memory descriptor byte layout (leading/stride-dimension
// byte offsets) is derived independently from the PTX ISA's canonical
// K-major layout formula, not copied from that reference, because the
// reference does not validate numerical correctness and its per-operand
// leading/stride-dimension values could not be reconciled with a K-major
// interpretation.
//
// Exit codes: 0 = success (or --help/--self-test pass), 1 =
// validation/CUDA/self-test failure, 2 = command-line usage error.
//
// This binary never selects a GPU, never reads any environment variable
// except EXPECTED_GPU_UUID (set by scripts/run_gpu.sh), and requires
// exactly one visible device at compute capability 10.3.

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
#include <cuda/ptx>  // cuda::ptx:: low-level PTX wrappers (mbarrier, fence_proxy_async, elect_sync).

namespace {

// ---------------------------------------------------------------------------
// Frozen contract constants.
// ---------------------------------------------------------------------------
constexpr int kThreadsPerCta = 128;
constexpr int kGridBlocks = 1;
constexpr int kCtaGroup = 1;
constexpr int kM = 128;
constexpr int kK = 16;
constexpr int kComputeCapabilityMajor = 10;
constexpr int kComputeCapabilityMinor = 3;
constexpr const char* kSchemaVersion = "1";
constexpr const char* kMethodName = "umma_1sm";
constexpr const char* kOperandPath = "smem_smem";
constexpr const char* kInputType = "bf16";
constexpr const char* kAccumulatorType = "fp32";

// TimingMode: propagated as an explicit kernel argument so that
// --self-test, pre-timing correctness validation, and warm-up
// launches never execute a %clock64 read, while only genuinely timed
// repetitions do. The mode is uniform across the whole grid (a launch
// argument, not a per-thread value), so branching on it is not divergent.
enum class TimingMode : int32_t { kUntimed = 0, kTimed = 1 };

int g_cleanup_failures = 0;

[[noreturn]] void fail(const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "umma_1sm: ERROR: ");
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

// ---------------------------------------------------------------------------
// Checked 64-bit integer arithmetic for FLOP/UMMA accounting.
// Host-only, so the __int128 GCC/Clang extension (available in nvcc's host
// compilation pass on the pinned Linux toolchain) gives a simple, obviously
// correct overflow check without hand-rolled sign-case branching.
// ---------------------------------------------------------------------------
int64_t checked_mul_i64(int64_t a, int64_t b, const char* what) {
    const __int128 result = static_cast<__int128>(a) * static_cast<__int128>(b);
    if (result > static_cast<__int128>(INT64_MAX) || result < static_cast<__int128>(INT64_MIN)) {
        fail("integer overflow computing %s (a=%lld b=%lld)", what, (long long)a, (long long)b);
    }
    return static_cast<int64_t>(result);
}

// ---------------------------------------------------------------------------
// Device-side shared-memory layout: a single fixed, non-swizzled, K-major
// packing for both A (MxK) and B (KxN). Derivation:
//
// PTX ISA 9.3 Table 44 fixes K=16 for .kind::f16 dense .cta_group::1. With
// BF16 (16-bit) elements, T = 128/16 = 8 (the "128-bit normalization" factor
// from PTX ISA 9.3 section 9.7.17.3.3). Since K (=16) exactly equals 2*T
// (=16), the whole K extent fits in one elementary K-major "core tile" of 8
// rows/cols x 16 K-elements (two T=8 chunks), per the canonical K-major
// layout formula "((8,m),(T,2k)):((1T,SBO),(1,LBO))" (PTX ISA 9.3 section
// 9.7.17.3.3). Each 8x16 core tile is packed row-major (contiguous) into 64
// BF16 elements; core tiles are then placed back-to-back with no gaps:
//   LBO (stride between the tile's two 8-element K-chunks) = 128 bytes (64 BF16 elements)
//   SBO (stride between successive 8-row/col groups)       = 256 bytes (128 BF16 elements)
// This gives a fully contiguous SMEM buffer, matching A_bytes = M*K*2 and
// B_bytes = N*K*2 exactly, with no padding.
// ---------------------------------------------------------------------------

// smem_core_tile_index: maps a logical (group_idx, pos_in_group, k) triple
// to a flat BF16-element offset within the fully-contiguous per-operand SMEM
// buffer, per the derivation above. group_idx is row/8 (for A) or col/8 (for
// B); pos_in_group is row%8 (for A) or col%8 (for B).
__device__ __forceinline__ int smem_core_tile_index(int group_idx, int pos_in_group, int k) {
    constexpr int kSboElem = 128;  // 8 rows * 16 K-elements
    constexpr int kLboElem = 64;   // 8 rows * 8 (T) K-elements
    constexpr int kT = 8;
    const int chunk = k / kT;
    const int t = k % kT;
    return group_idx * kSboElem + chunk * kLboElem + pos_in_group * kT + t;
}

// make_smem_descriptor: builds the 64-bit shared memory descriptor (PTX ISA
// 9.3 Table 45) for a matrix stored at smem_ptr using the fixed layout above.
// LBO/SBO are constants shared by every specialization (they depend only on
// K=16 and BF16's T=8, not on M or N); only the base address differs between
// the A and B descriptors.
__device__ __forceinline__ uint64_t make_smem_descriptor(const void* smem_ptr) {
    constexpr uint32_t kLboBytes = 128;  // 64 elements * 2 bytes/element
    constexpr uint32_t kSboBytes = 256;  // 128 elements * 2 bytes/element
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    // Bits 0-13: matrix-descriptor-encode(matrix start address).
    desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFFu);
    // Bits 14-15: reserved, must be 0 (left unset).
    // Bits 16-29: matrix-descriptor-encode(leading dimension byte offset).
    desc |= static_cast<uint64_t>((kLboBytes >> 4) & 0x3FFFu) << 16;
    // Bits 30-31: reserved, must be 0 (left unset).
    // Bits 32-45: matrix-descriptor-encode(stride dimension byte offset).
    desc |= static_cast<uint64_t>((kSboBytes >> 4) & 0x3FFFu) << 32;
    // Bits 46-48: fixed constant 0b001.
    desc |= (UINT64_C(1) << 46);
    // Bits 49-51: matrix base offset = 0 (no swizzling selected).
    // Bit 52: leading-dimension stride mode = 0 (relative byte offset).
    // Bits 53-60: fixed constant, all zero.
    // Bits 61-63: swizzling mode = 0 (no swizzling).
    return desc;
}

// ---------------------------------------------------------------------------
// Instruction descriptor (PTX ISA 9.3 Table 47, .kind::f16 column): dense
// (sparsity=0), no integer saturation (n/a for this kind), dtype=FP32(1),
// atype=BF16(1), btype=BF16(1), no negate, no transpose, N = encoded N>>3,
// M = encoded M>>4 (M is fixed at 128 for every specialization), no
// .ws B-matrix reuse shift. Every reserved bit is left at 0.
// ---------------------------------------------------------------------------
template <int N>
__device__ constexpr uint32_t make_instruction_descriptor() {
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128, or 256");
    static_assert(((static_cast<uint32_t>(N) >> 3) & ~0x3Fu) == 0, "N field overflows bits 17-22");
    uint32_t desc = 0;
    desc |= (1u << 4);                                     // bits 4-5: dtype = FP32
    desc |= (1u << 7);                                      // bits 7-9: atype = BF16
    desc |= (1u << 10);                                     // bits 10-12: btype = BF16
    desc |= ((static_cast<uint32_t>(N) >> 3) << 17);         // bits 17-22: N >> 3
    desc |= ((static_cast<uint32_t>(kM) >> 4) << 24);        // bits 24-28: M >> 4
    return desc;
}

// validate_instruction_descriptor: field-by-field re-derivation of
// make_instruction_descriptor's output, independent of that function's own
// shift/mask expressions, so a regression in either one fails to compile.
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
           m_field == (static_cast<uint32_t>(kM) >> 4) && reserved29 == 0 && ws_shift == 0;
}

static_assert(validate_instruction_descriptor<64>(), "N=64 instruction descriptor is malformed");
static_assert(validate_instruction_descriptor<128>(), "N=128 instruction descriptor is malformed");
static_assert(validate_instruction_descriptor<256>(), "N=256 instruction descriptor is malformed");

// ---------------------------------------------------------------------------
// tcgen05 inline-PTX primitives. CUDA 13.1's <cuda/ptx> header does not wrap
// the tcgen05 family (Blackwell-only, introduced PTX ISA 8.6), so these are
// hand-written from the syntax in PTX ISA 9.3 sections 9.7.17.7 (alloc/
// dealloc/relinquish), 9.7.17.10.9.1 (mma), and 9.7.17.12.1 (commit). The
// generic mbarrier/fence/elect operations reuse cuda::ptx:: wrappers, the
// same ones used by the TMA copy microbenchmark.
// ---------------------------------------------------------------------------

// One tcgen05.mma.cta_group::1.kind::f16 issue. enable_input_d is passed as
// a plain 0/1 register and converted to a PTX predicate inside the asm
// block, matching the documented idesc/enable-input-d operand order (PTX ISA
// 9.3 section 9.7.17.10.9.1, syntax form 1); disable-output-lane is omitted,
// which the same section's worked example
// ("tcgen05.mma.cta_group.kind [taddr0], adesc, bdesc, idesc, p;" in section
// 9.7.17.6.4.2) shows is valid.
__device__ __forceinline__ void issue_one_umma(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                                uint32_t idesc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p_enable_d;\n\t"
        "setp.ne.b32 p_enable_d, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_enable_d;\n\t"
        "}\n\t"
        :
        : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
        : "memory");
}

// tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64, tracking the
// completion of every asynchronous tcgen05.mma issued so far by this thread
// (PTX ISA 9.3 section 9.7.17.12.1). No cross-CTA signaling qualifier is
// used: this kernel never declares a cluster (single CTA, cta_group::1
// throughout), and such qualifiers are reserved for the 2-SM arm.
__device__ __forceinline__ void commit_umma(uint32_t mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" : : "r"(mbar_addr) : "memory");
}

// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32: issued
// collectively by every thread of one warp (Issue Granularity, PTX ISA 9.3
// Table 51). dst_smem_addr is a shared-memory address that receives the
// allocated Tensor Memory address.
__device__ __forceinline__ void tcgen05_alloc(uint32_t dst_smem_addr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(dst_smem_addr), "r"(n_cols)
                 : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" : : "r"(taddr), "r"(n_cols) : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" : : : "memory");
}

// tcgen05.ld.sync.aligned.32x32b.x32.b32: a warp-collective load of one
// 32-lane x 32-column fragment (PTX ISA 9.3 Table 54 and section
// 9.7.17.8.3). All 32 threads of the issuing warp must pass the same
// taddr. Its bits 31-16 select the first physical TMEM lane, and the access
// restrictions assign each warp a different 32-lane chunk (PTX ISA 9.3
// section 9.7.17.8.1), so the caller must encode that warp-specific lane
// base in taddr.
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

// ---------------------------------------------------------------------------
// TMEM load address construction (PTX ISA 9.3 section 9.7.17.1.1, "Tensor
// Memory Addressing"): a 32-bit TMEM address packs a lane index in bits 31-16
// and a column index in bits 15-0. Section 9.7.17.8.1, "Access restrictions",
// further requires that the Tensor Memory of a CTA be split into four
// 32-lane chunks, one per warp of the warpgroup (warp 0 -> lanes 0-31, warp 1
// -> lanes 32-63, warp 2 -> lanes 64-95, warp 3 -> lanes 96-127), and that
// every thread of the issuing warp supply the identical collective taddr.
// tmem_base (the address returned by tcgen05.alloc) is always lane 0 of the
// allocated columns, so each warp must add its own lane contribution before
// issuing tcgen05.ld; omitting it would make every warp read warp 0's lanes
// instead of its own.
// ---------------------------------------------------------------------------
constexpr uint32_t kTmemLaneShift = 16;        // PTX ISA 9.3 9.7.17.1.1: lane index occupies bits 31-16.
constexpr uint32_t kTmemRowsPerWarp = 32;      // PTX ISA 9.3 9.7.17.8.1: each warp owns a 32-lane chunk.
constexpr uint32_t kTmemColsPerFragment = 32;  // This kernel's fixed 32-column tcgen05.ld.x32 fragment width.

// make_tmem_load_address: the collective taddr passed to
// tcgen05_ld_32x32b_x32 for warp warp_id's fragment frag, including that
// warp's own lane contribution.
__device__ __forceinline__ uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {
    const uint32_t lane_contribution = (static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift;
    const uint32_t column_contribution = static_cast<uint32_t>(frag) * kTmemColsPerFragment;
    return tmem_base + lane_contribution + column_contribution;
}

// ---------------------------------------------------------------------------
// Launch contract: every visible kernel must reject any launch that is not
// exactly grid=(1,1,1), block=(128,1,1). __launch_bounds__(128) only caps the
// maximum thread count, so this is an explicit guard evaluated before any
// __syncthreads(), mbarrier initialization, TMEM allocation, or UMMA
// instruction (see the top of umma_1sm_body below), so a rejected launch can
// never leave TMEM allocated or block on a barrier.
// ---------------------------------------------------------------------------
constexpr int kExpectedGridDim = kGridBlocks;
constexpr int kExpectedBlockDimX = kThreadsPerCta;

__device__ __forceinline__ bool launch_contract_is_valid() {
    return gridDim.x == kExpectedGridDim && gridDim.y == kExpectedGridDim && gridDim.z == kExpectedGridDim &&
           blockDim.x == kExpectedBlockDimX && blockDim.y == 1 && blockDim.z == 1;
}

// ---------------------------------------------------------------------------
// Device: templated kernel body, instantiated once per (N, DEPTH)
// specialization by the extern "C" wrappers below: stable extern "C"
// symbols share one templated, force-inlined body, so each symbol still
// keeps its own SASS specialization.
// ---------------------------------------------------------------------------
template <int N, int DEPTH>
__device__ __forceinline__ void umma_1sm_body(int64_t iterations, TimingMode timing_mode,
                                               float* __restrict__ g_d_out,
                                               unsigned long long* __restrict__ g_elapsed_cycles,
                                               int* __restrict__ g_launch_ok) {
    static_assert(N == 64 || N == 128 || N == 256, "N must be 64, 128, or 256");
    static_assert(DEPTH == 4 || DEPTH == 16 || DEPTH == 64 || DEPTH == 256,
                  "DEPTH must be 4, 16, 64, or 256");

    const int tid = threadIdx.x;

    // ---- Launch contract: must precede any __syncthreads(), mbarrier init, ----
    // ---- TMEM allocation, or UMMA instruction (see launch_contract_is_valid --
    // ---- above). A rejected launch writes 0 and returns immediately, before --
    // ---- touching any shared state, so it can never allocate TMEM or block --
    // ---- on a barrier. A silent early return alone would not be observable --
    // ---- by the host, so every accepted launch also confirms itself. --------
    if (!launch_contract_is_valid()) {
        if (tid == 0) {
            g_launch_ok[0] = 0;
        }
        return;
    }
    if (tid == 0) {
        g_launch_ok[0] = 1;
    }

    constexpr int kABytes = kM * kK * 2;  // BF16 = 2 bytes/element

    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B = reinterpret_cast<__nv_bfloat16*>(smem + kABytes);

    const int warp_id = tid / 32;

    // ---- Fill A and B with the frozen validation pattern, ---------------
    // ---- placed directly into the fixed K-major physical layout. --------
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

    // Gather every thread's A/B writes before the single fencing thread's
    // fence.proxy.async, so that fence also publishes those writes (not
    // just its own) to the async proxy (PTX ISA 9.3 section 9.7.17.6.5).
    __syncthreads();

    if (tid == 0) {
        cuda::ptx::mbarrier_init(&mbar, 1u);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    // Publish the mbarrier initialization and the fence's effect to every
    // thread, including whichever thread elect_sync selects as leader below.
    __syncthreads();

    bool is_leader = false;
    if (tid < 32) {
        is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    }

    // TMEM allocation: issued collectively by every thread of warp 0 (Issue
    // Granularity, PTX ISA 9.3 Table 51), exactly N columns.
    if (warp_id == 0) {
        const uint32_t tmem_addr_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared));
        tcgen05_alloc(tmem_addr_smem, static_cast<uint32_t>(N));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_instruction_descriptor<N>();

    // ---- Timed region: leader thread only. -----------------
    unsigned long long elapsed_cycles = 0;
    if (is_leader) {
        const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        uint64_t start_clock = 0, end_clock = 0;
        uint32_t parity = 0;

        if (timing_mode == TimingMode::kTimed) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock) : : "memory");
        }
        for (int64_t it = 0; it < iterations; ++it) {
            issue_one_umma(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/0);
#pragma unroll
            for (int m = 1; m < DEPTH; ++m) {
                issue_one_umma(tmem_d, a_desc, b_desc, idesc, /*enable_input_d=*/1);
            }
            commit_umma(mbar_addr);
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {
            }
            parity ^= 1u;
        }
        if (timing_mode == TimingMode::kTimed) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock) : : "memory");
            elapsed_cycles = static_cast<unsigned long long>(end_clock - start_clock);
        }
        g_elapsed_cycles[0] = elapsed_cycles;
    }

    // ---- Untimed readback: every thread. -----------------
    // The leader's confirmed mbarrier completion, followed by this barrier
    // (an "execution ordering operation" per PTX ISA 9.3 section 9.7.17.6.3),
    // establishes the cross-thread ordering the canonical "non-pipelined,
    // different thread" pattern (section 9.7.17.6.4.4) requires before any
    // other thread's fence::after_thread_sync + tcgen05.ld.
    __syncthreads();
    tcgen05_fence_after_thread_sync();

    constexpr int kFragments = N / 32;
    static_assert(kFragments * 32 == N, "N must be a multiple of 32 for 32-column fragments");
    const int lane = tid % 32;
    const int row = warp_id * 32 + lane;
#pragma unroll
    for (int frag = 0; frag < kFragments; ++frag) {
        uint32_t regs[32];
        tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);
        tcgen05_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            g_d_out[row * N + frag * 32 + i] = __uint_as_float(regs[i]);
        }
    }

    // ---- TMEM lifecycle close-out. -----------------
    __syncthreads();
    if (warp_id == 0) {
        tcgen05_dealloc(tmem_d, static_cast<uint32_t>(N));
        tcgen05_relinquish_alloc_permit();
    }
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     :
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar)))
                     : "memory");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The twelve extern "C" specializations.
// ---------------------------------------------------------------------------
#define UMMA_1SM_DEFINE_KERNEL(N, DEPTH)                                                     \
    extern "C" __global__ __launch_bounds__(128) void umma_1sm_m128n##N##k16_d##DEPTH(        \
        int64_t iterations, TimingMode timing_mode, float* g_d_out,                           \
        unsigned long long* g_elapsed_cycles, int* g_launch_ok) {                             \
        umma_1sm_body<N, DEPTH>(iterations, timing_mode, g_d_out, g_elapsed_cycles, g_launch_ok); \
    }

UMMA_1SM_DEFINE_KERNEL(64, 4)
UMMA_1SM_DEFINE_KERNEL(64, 16)
UMMA_1SM_DEFINE_KERNEL(64, 64)
UMMA_1SM_DEFINE_KERNEL(64, 256)
UMMA_1SM_DEFINE_KERNEL(128, 4)
UMMA_1SM_DEFINE_KERNEL(128, 16)
UMMA_1SM_DEFINE_KERNEL(128, 64)
UMMA_1SM_DEFINE_KERNEL(128, 256)
UMMA_1SM_DEFINE_KERNEL(256, 4)
UMMA_1SM_DEFINE_KERNEL(256, 16)
UMMA_1SM_DEFINE_KERNEL(256, 64)
UMMA_1SM_DEFINE_KERNEL(256, 256)

#undef UMMA_1SM_DEFINE_KERNEL

// ---------------------------------------------------------------------------
// Host: CPU reference, device query, CLI, CSV, orchestration.
// ---------------------------------------------------------------------------
namespace {

typedef void (*KernelFn)(int64_t, TimingMode, float*, unsigned long long*, int*);

struct Specialization {
    int n = 0;
    int depth = 0;
    KernelFn kernel = nullptr;
    const char* symbol = "";
};

#define UMMA_1SM_SPEC_ENTRY(N, DEPTH) \
    { N, DEPTH, &umma_1sm_m128n##N##k16_d##DEPTH, "umma_1sm_m128n" #N "k16_d" #DEPTH }

constexpr Specialization kSpecializations[12] = {
    UMMA_1SM_SPEC_ENTRY(64, 4),   UMMA_1SM_SPEC_ENTRY(64, 16),   UMMA_1SM_SPEC_ENTRY(64, 64),
    UMMA_1SM_SPEC_ENTRY(64, 256), UMMA_1SM_SPEC_ENTRY(128, 4),   UMMA_1SM_SPEC_ENTRY(128, 16),
    UMMA_1SM_SPEC_ENTRY(128, 64), UMMA_1SM_SPEC_ENTRY(128, 256), UMMA_1SM_SPEC_ENTRY(256, 4),
    UMMA_1SM_SPEC_ENTRY(256, 16), UMMA_1SM_SPEC_ENTRY(256, 64),  UMMA_1SM_SPEC_ENTRY(256, 256),
};

#undef UMMA_1SM_SPEC_ENTRY

const Specialization* find_spec(int n, int depth) {
    for (const auto& s : kSpecializations) {
        if (s.n == n && s.depth == depth) return &s;
    }
    return nullptr;
}

// ---- CPU reference: logical formulas only, independent of the -------------
// ---- GPU kernel's physical SMEM packing. -------------------------------
int32_t reference_a(int row, int k) { return ((row + 3 * k) % 7) - 3; }
int32_t reference_b(int k, int col) { return ((2 * k + col) % 5) - 2; }

double reference_d(int row, int col, int depth) {
    int64_t sum = 0;
    for (int k = 0; k < kK; ++k) {
        sum += static_cast<int64_t>(reference_a(row, k)) * static_cast<int64_t>(reference_b(k, col));
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
    for (int row = 0; row < kM; ++row) {
        for (int col = 0; col < n; ++col) {
            const double expected = reference_d(row, col, depth);
            const double obtained = static_cast<double>(d_out[static_cast<size_t>(row) * n + col]);
            const double abs_error = std::fabs(obtained - expected);
            if (abs_error > r.max_abs_error) r.max_abs_error = abs_error;
            if (obtained != expected) {
                if (r.mismatches == 0) {
                    r.first_mismatch_index = static_cast<int64_t>(row) * n + col;
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

// ---------------------------------------------------------------------------
// GPU environment/provenance: exactly one visible device,
// logical device 0, compute capability 10.3, and its CUDA-reported UUID
// (never nvidia-smi) matches EXPECTED_GPU_UUID.
// ---------------------------------------------------------------------------
struct GpuInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    std::string uuid;
    int driver_version = 0;
    int runtime_version = 0;
};

std::string format_gpu_uuid(const cudaUUID_t& uuid) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0],
                  b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
                  b[14], b[15]);
    return std::string(buf);
}

bool looks_like_gpu_uuid(const std::string& s) {
    // "GPU-" + 8-4-4-4-12 hex digits (36 chars incl. 4 dashes) = 40 chars total.
    if (s.size() != 40) return false;
    if (s.compare(0, 4, "GPU-") != 0) return false;
    for (size_t i = 4; i < s.size(); ++i) {
        const size_t j = i - 4;  // position within the 36-char UUID body
        if (j == 8 || j == 13 || j == 18 || j == 23) {
            if (s[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(s[i]))) {
            return false;
        }
    }
    return true;
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

GpuInfo query_and_verify_gpu() {
    int device_count = 0;
    CUDA_CHECK_FATAL(cudaGetDeviceCount(&device_count));
    if (device_count != 1) {
        fail("expected exactly 1 visible CUDA device, found %d", device_count);
    }
    CUDA_CHECK_FATAL(cudaSetDevice(0));
    int current_device = -1;
    CUDA_CHECK_FATAL(cudaGetDevice(&current_device));
    if (current_device != 0) {
        fail("expected the visible device to be logical device 0, got %d", current_device);
    }

    cudaDeviceProp prop{};
    CUDA_CHECK_FATAL(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != kComputeCapabilityMajor || prop.minor != kComputeCapabilityMinor) {
        fail("expected compute capability %d.%d, found %d.%d", kComputeCapabilityMajor,
             kComputeCapabilityMinor, prop.major, prop.minor);
    }

    const char* uuid_env = std::getenv("EXPECTED_GPU_UUID");
    if (uuid_env == nullptr || uuid_env[0] == '\0') {
        fail("EXPECTED_GPU_UUID is not set; run this binary only via scripts/run_gpu.sh");
    }
    const std::string expected_uuid(uuid_env);
    if (!looks_like_gpu_uuid(expected_uuid)) {
        fail("EXPECTED_GPU_UUID='%s' is not correctly formatted (expected GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)",
             expected_uuid.c_str());
    }
    const std::string visible_uuid = format_gpu_uuid(prop.uuid);
    if (to_lower(visible_uuid) != to_lower(expected_uuid)) {
        fail("visible GPU UUID %s does not match EXPECTED_GPU_UUID %s", visible_uuid.c_str(),
             expected_uuid.c_str());
    }

    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK_FATAL(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK_FATAL(cudaRuntimeGetVersion(&runtime_version));

    GpuInfo info;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.uuid = visible_uuid;
    info.driver_version = driver_version;
    info.runtime_version = runtime_version;
    return info;
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------
struct CliConfig {
    bool help = false;
    bool self_test = false;
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
    std::fprintf(
        out,
        "umma_1sm - 1-SM BF16 UMMA (tcgen05.mma, kind::f16, cta_group::1) instruction microbenchmark\n"
        "\n"
        "1-SM arm of the BF16 UMMA throughput experiment. Issues depth-many\n"
        "tcgen05.mma instructions per iteration from a single CTA and reports\n"
        "elapsed cycles. Not a TFLOP/s or saturation claim: this measures\n"
        "instruction issue, not sustained tensor-core throughput.\n"
        "\n"
        "Usage:\n"
        "  umma_1sm --help\n"
        "  umma_1sm --self-test\n"
        "  umma_1sm --run-kind {smoke,benchmark} --n {64,128,256} --depth {4,16,64,256} \\\n"
        "           --iterations N --warmup-iterations N --repetitions N\n"
        "\n"
        "Options:\n"
        "  --run-kind {smoke,benchmark}  Required. Labels the CSV row; both kinds run\n"
        "                                identically. Neither is a publishable result.\n"
        "  --n {64,128,256}              Required. N dimension of matrix B / D.\n"
        "  --depth {4,16,64,256}         Required. tcgen05.mma instructions per\n"
        "                                iteration, fully unrolled at compile time.\n"
        "  --iterations N                Required, N >= 1. Timed outer-loop repeats\n"
        "                                per kernel launch.\n"
        "  --warmup-iterations N         Required, N >= 0. Untimed kernel launches\n"
        "                                before the timed repetitions.\n"
        "  --repetitions N               Required, N >= 1. Separately timed kernel\n"
        "                                launches; one CSV row each.\n"
        "  --self-test                   Validate all twelve specializations on a\n"
        "                                small fixed iteration count and exit; no CSV,\n"
        "                                no timing. Cannot combine with other options.\n"
        "  --help                        Show this help and exit.\n"
        "\n"
        "On a --run-kind run, stdout carries only CSV (one header line plus one row\n"
        "per repetition); diagnostics, progress, and errors go to stderr. See\n"
        "README.md for the CSV schema and units.\n");
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

        if (arg == "--help") {
            if (cfg->help) { *err = "--help specified more than once"; return false; }
            cfg->help = true;
            continue;
        }
        if (arg == "--self-test") {
            if (cfg->self_test) { *err = "--self-test specified more than once"; return false; }
            cfg->self_test = true;
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

    if (cfg->help && cfg->self_test) {
        *err = "--help and --self-test are mutually exclusive";
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
    if (cfg->self_test) {
        if (cfg->has_run_kind || cfg->has_n || cfg->has_depth || cfg->has_iterations ||
            cfg->has_warmup_iterations || cfg->has_repetitions) {
            *err = "--self-test cannot be combined with other options";
            return false;
        }
        return true;
    }

    if (!cfg->has_run_kind) { *err = "--run-kind is required (unless --help/--self-test)"; return false; }
    if (!cfg->has_n) { *err = "--n is required (unless --help/--self-test)"; return false; }
    if (!cfg->has_depth) { *err = "--depth is required (unless --help/--self-test)"; return false; }
    if (!cfg->has_iterations) { *err = "--iterations is required (unless --help/--self-test)"; return false; }
    if (!cfg->has_warmup_iterations) {
        *err = "--warmup-iterations is required (unless --help/--self-test)";
        return false;
    }
    if (!cfg->has_repetitions) { *err = "--repetitions is required (unless --help/--self-test)"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// CSV: frozen header and column order.
// ---------------------------------------------------------------------------
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
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) any = true;
    const int rc = pclose(pipe);
    if (rc != 0) return "unknown";
    return any ? "true" : "false";
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
    GpuInfo gpu;
    std::string git_commit;
    std::string git_dirty;
};

void print_csv_header() {
    std::printf(
        "schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,"
        "depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,"
        "flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,"
        "grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,"
        "max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,"
        "cuda_runtime_version,git_commit,git_dirty\n");
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
    oss << kSchemaVersion << ',' << r.timestamp_utc << ',' << r.run_kind << ',' << "false" << ','
        << kMethodName << ',' << r.sample_index << ',' << kCtaGroup << ',' << kM << ',' << r.n << ','
        << kK << ',' << r.depth << ',' << r.iterations << ',' << r.warmup_iterations << ','
        << r.repetitions << ',' << r.depth << ',' << r.total_umma << ',' << r.flops_per_umma << ','
        << r.total_flops << ',' << r.elapsed_cycles << ',' << cycles_per_umma << ','
        << flops_per_cycle << ',' << kThreadsPerCta << ',' << kGridBlocks << ',' << r.n << ','
        << kOperandPath << ',' << kInputType << ',' << kAccumulatorType << ',' << "OK" << ',' << 0
        << ',' << 0 << ',' << csv_quote(r.gpu.name) << ',' << r.gpu.uuid << ',' << r.gpu.major << '.'
        << r.gpu.minor << ',' << r.gpu.driver_version << ',' << r.gpu.runtime_version << ','
        << r.git_commit << ',' << r.git_dirty << '\n';
    std::fputs(oss.str().c_str(), stdout);
}

// ---------------------------------------------------------------------------
// Orchestration: one specialization, one kernel launch + readback + host
// validation, shared by the main run path and --self-test.
// ---------------------------------------------------------------------------
struct RunResult {
    ValidationResult validation;
    unsigned long long elapsed_cycles = 0;
};

RunResult run_once(const Specialization& spec, int64_t iterations, TimingMode mode) {
    RunResult result;
    const size_t d_elems = static_cast<size_t>(kM) * static_cast<size_t>(spec.n);
    float* d_out_device = nullptr;
    unsigned long long* cycles_device = nullptr;
    int* launch_ok_device = nullptr;
    CUDA_CHECK_FATAL(cudaMalloc(&d_out_device, d_elems * sizeof(float)));
    CUDA_CHECK_FATAL(cudaMalloc(&cycles_device, sizeof(unsigned long long)));
    CUDA_CHECK_FATAL(cudaMalloc(&launch_ok_device, sizeof(int)));
    CUDA_CHECK_FATAL(cudaMemset(cycles_device, 0, sizeof(unsigned long long)));
    // 0 means "not confirmed": both an explicit device-side rejection and a
    // kernel that never ran at all collapse to this same not-OK value.
    CUDA_CHECK_FATAL(cudaMemset(launch_ok_device, 0, sizeof(int)));

    const int smem_bytes = kM * kK * 2 + spec.n * kK * 2;
    spec.kernel<<<kGridBlocks, kThreadsPerCta, static_cast<size_t>(smem_bytes)>>>(
        iterations, mode, d_out_device, cycles_device, launch_ok_device);
    CUDA_CHECK_FATAL(cudaGetLastError());
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());

    int launch_ok_host = 0;
    CUDA_CHECK_FATAL(cudaMemcpy(&launch_ok_host, launch_ok_device, sizeof(launch_ok_host),
                                 cudaMemcpyDeviceToHost));
    std::vector<float> d_out_host(d_elems);
    CUDA_CHECK_FATAL(
        cudaMemcpy(d_out_host.data(), d_out_device, d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FATAL(cudaMemcpy(&result.elapsed_cycles, cycles_device, sizeof(result.elapsed_cycles),
                                 cudaMemcpyDeviceToHost));
    CUDA_CHECK_FATAL(cudaFree(d_out_device));
    CUDA_CHECK_FATAL(cudaFree(cycles_device));
    CUDA_CHECK_FATAL(cudaFree(launch_ok_device));

    // The host launcher below always requests grid=(1,1,1) block=(128,1,1),
    // so this can only fire if the launch contract check itself regresses;
    // it must still abort loudly rather than silently validating garbage D.
    if (launch_ok_host != 1) {
        fail("launch-contract violation for %s: kernel did not confirm grid=(1,1,1) "
             "block=(128,1,1) (launch_ok=%d)",
             spec.symbol, launch_ok_host);
    }

    result.validation = validate_d(d_out_host, spec.n, spec.depth);
    return result;
}

// Reports a correctness mismatch and terminates with a nonzero exit code:
// a numerical error must prevent any subsequent timing or CSV output, so
// every caller on the main run path needs exactly this abort-on-failure
// behavior.
[[noreturn]] void die_on_validation_failure(const Specialization& spec, const ValidationResult& validation) {
    std::fprintf(stderr,
                 "umma_1sm: ERROR: correctness validation FAILED for %s: mismatches=%lld "
                 "first_index=%lld expected=%.1f obtained=%.1f max_abs_error=%.6g\n",
                 spec.symbol, (long long)validation.mismatches, (long long)validation.first_mismatch_index,
                 validation.first_mismatch_expected, validation.first_mismatch_obtained,
                 validation.max_abs_error);
    fail("correctness validation failed for %s; no timing or CSV output was produced", spec.symbol);
}

// Runs one specialization untimed (timing_mode=kUntimed, so neither %clock64
// read executes) and aborts on any correctness mismatch. Used
// for pre-timing validation and warm-up, neither of which may be timed.
void run_untimed_or_die(const Specialization& spec, int64_t iterations) {
    const RunResult result = run_once(spec, iterations, TimingMode::kUntimed);
    if (!result.validation.ok) die_on_validation_failure(spec, result.validation);
}

// Runs one specialization timed (timing_mode=kTimed) and aborts on any
// correctness mismatch or on an unexpectedly-zero cycle count. Used only for
// genuinely timed repetitions.
unsigned long long run_timed_or_die(const Specialization& spec, int64_t iterations) {
    const RunResult result = run_once(spec, iterations, TimingMode::kTimed);
    if (!result.validation.ok) die_on_validation_failure(spec, result.validation);
    if (result.elapsed_cycles == 0) {
        fail("internal error: elapsed_cycles was not greater than zero for %s", spec.symbol);
    }
    return result.elapsed_cycles;
}

constexpr int64_t kSelfTestIterations = 2;

int run_self_test() {
    std::fprintf(stderr, "umma_1sm: SELF_TEST start\n");
    int passed = 0;
    for (const auto& spec : kSpecializations) {
        const RunResult result = run_once(spec, kSelfTestIterations, TimingMode::kUntimed);
        std::fprintf(stderr,
                     "umma_1sm: SELF_TEST %s n=%d depth=%d result=%s mismatches=%lld "
                     "max_abs_error=%.6g\n",
                     spec.symbol, spec.n, spec.depth, result.validation.ok ? "PASS" : "FAIL",
                     (long long)result.validation.mismatches, result.validation.max_abs_error);
        if (result.validation.ok) ++passed;
    }
    std::fprintf(stderr, "umma_1sm: SELF_TEST_RESULT %d/%d\n", passed, 12);
    if (passed == 12) {
        std::fprintf(stdout, "SELF_TEST: PASS (12/12)\n");
        return 0;
    }
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig cli;
    std::string parse_err;
    if (!parse_cli(argc, argv, &cli, &parse_err)) {
        std::fprintf(stderr, "umma_1sm: ERROR: %s\n", parse_err.c_str());
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
    const GpuInfo gpu = query_and_verify_gpu();

    if (cli.self_test) {
        return run_self_test();
    }

    const Specialization* spec = find_spec(cli.n, cli.depth);
    if (spec == nullptr) {
        fail("internal error: no specialization for n=%d depth=%d", cli.n, cli.depth);
    }

    std::fprintf(stderr,
                 "umma_1sm: run_kind=%s n=%d depth=%d iterations=%lld warmup_iterations=%lld "
                 "repetitions=%lld\n",
                 cli.run_kind.c_str(), cli.n, cli.depth, (long long)cli.iterations,
                 (long long)cli.warmup_iterations, (long long)cli.repetitions);

    // Step 1: validate without timing before anything else.
    run_untimed_or_die(*spec, cli.iterations);
    std::fprintf(stderr, "umma_1sm: correctness=OK mismatches=0 (pre-timing check)\n");

    // Step 2: warm-up, discarded, also untimed.
    for (int64_t w = 0; w < cli.warmup_iterations; ++w) {
        run_untimed_or_die(*spec, cli.iterations);
    }

    // Steps 3-5: timed repetitions, each re-validated before its CSV row.
    print_csv_header();
    const int64_t flops_per_umma =
        checked_mul_i64(checked_mul_i64(2, kM, "2*M"), checked_mul_i64(spec->n, kK, "N*K"), "flops_per_umma");
    const int64_t total_umma = checked_mul_i64(spec->depth, cli.iterations, "depth*iterations");
    const int64_t total_flops = checked_mul_i64(flops_per_umma, total_umma, "flops_per_umma*total_umma");

    const std::string git_commit_str = git_commit_hash();
    const std::string git_dirty_str = git_dirty_flag();

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
        row.gpu = gpu;
        row.git_commit = git_commit_str;
        row.git_dirty = git_dirty_str;
        print_csv_row(row);
    }

    if (g_cleanup_failures != 0) {
        std::fprintf(stderr, "umma_1sm: ERROR: %d resource cleanup failure(s) occurred\n",
                     g_cleanup_failures);
        return 1;
    }
    return 0;
}
