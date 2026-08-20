// 2-SM BF16 UMMA instruction microbenchmark: the CTA-pair, cta_group::2 arm
// of the "BF16 UMMA throughput" experiment. It shares no code with the 1-SM
// arm; every descriptor, primitive, and synchronization step is re-derived
// here for the CTA-pair case.
//
// Frozen experimental contract: tcgen05.mma, kind::f16,
// cta_group::2, BF16 x BF16 -> FP32, M=256 (128 local rows per CTA of a
// two-CTA cluster), K=16 (implied by kind::f16 dense BF16), N in
// {64,128,256}, depth in {4,16,64,256}, A and B in SMEM, D in TMEM, exactly
// one static two-CTA cluster of 128 threads per CTA, exactly twelve
// specializations.
//
// Primary normative source: NVIDIA PTX ISA 9.3
// (https://docs.nvidia.com/cuda/parallel-thread-execution/), chapter
// "9.7.17. TensorCore 5th Generation Family Instructions", read from the
// official PDF (https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf), including
// the CTA Pair / Issue Granularity section (9.7.17.5, 9.7.17.5.1), the M=256
// data-path layout (9.7.17.10.5, Figure 205/206, "Layout A"), and the
// tcgen05.commit multicast form (9.7.17.12.1). Secondary conceptual
// reference (adapted, not copied, and independently checked against the PTX
// ISA): the pinned revision of
// SemiAnalysisAI/microbench-blackwell/umma_throughput/umma_tput.cu. Every
// descriptor, synchronization step, TMEM address, and completion mechanism
// below was re-derived from the PTX ISA text and validated by compiling
// isolated probes against the pinned CUDA 13.1.80 toolchain before being
// assembled into this file.
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
#include <cuda/ptx>  // cuda::ptx:: low-level PTX wrappers (mbarrier, fence_proxy_async, elect_sync,
                     // get_sreg_cluster_ctarank/nctarank, barrier_cluster_arrive/wait).

namespace {

// ---------------------------------------------------------------------------
// Frozen contract constants.
// ---------------------------------------------------------------------------
constexpr int kThreadsPerCta = 128;
constexpr int kClusterCtas = 2;       // exactly one static two-CTA cluster
constexpr int kGridBlocks = 2;        // grid == cluster: exactly one cluster
constexpr int kCtaGroup = 2;
constexpr int kMGlobal = 256;         // joint M across the CTA pair (idesc/FLOP accounting only)
constexpr int kMLocal = 128;          // each CTA's own local A/D row count (physical SMEM/TMEM shape)
constexpr int kK = 16;
constexpr int kComputeCapabilityMajor = 10;
constexpr int kComputeCapabilityMinor = 3;
constexpr const char* kSchemaVersion = "1";
constexpr const char* kMethodName = "umma_2sm";
constexpr const char* kOperandPath = "smem_smem";
constexpr const char* kInputType = "bf16";
constexpr const char* kAccumulatorType = "fp32";

// TimingMode: propagated as an explicit kernel argument, so --self-test,
// pre-timing correctness validation, and warm-up launches never execute a
// %clock64 read while only genuinely timed repetitions do. The mode is
// uniform across the whole grid (a launch
// argument, not a per-thread value), so branching on it is not divergent.
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

// ---------------------------------------------------------------------------
// Checked 64-bit integer arithmetic for FLOP/UMMA accounting. Host-only.
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
// packing for both A (128 local rows x K) and B (K x N/2 per CTA). The
// formula depends only on K=16 and BF16's 128-bit-normalization factor T=8
// (PTX ISA 9.3 section 9.7.17.3.3); it does not depend on M or on the
// .cta_group qualifier, so each CTA's own local 128-row A tile and its local
// N/2-column B tile use exactly the same per-operand packing, re-derived
// here for M=256/cta_group::2 rather than carried over from the
// M=128/cta_group::1 case.
//
// LBO (stride between a tile's two 8-element K-chunks) = 128 bytes.
// SBO (stride between successive 8-row/col groups)     = 256 bytes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int smem_core_tile_index(int group_idx, int pos_in_group, int k) {
    constexpr int kSboElem = 128;  // 8 rows * 16 K-elements
    constexpr int kLboElem = 64;   // 8 rows * 8 (T) K-elements
    constexpr int kT = 8;
    const int chunk = k / kT;
    const int t = k % kT;
    return group_idx * kSboElem + chunk * kLboElem + pos_in_group * kT + t;
}

// make_smem_descriptor: builds the 64-bit shared memory descriptor (PTX ISA
// 9.3 Table 45) for a matrix stored at smem_ptr in the CALLING thread's own
// CTA. LBO/SBO are constants shared by every specialization; only the base
// address differs between the A and B descriptors. For cta_group::2, the
// CTA-pair hardware applies this same descriptor (and in particular its
// address, interpreted as a relative offset) to each CTA's own local shared
// memory bank, so A, B, the mbarrier, and the TMEM-address shared variable
// must occupy identical relative offsets in both CTAs for this to apply
// correctly to both halves of the joint operation.
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
// Instruction descriptor (PTX ISA 9.3 Table 47, .kind::f16 column -- the
// same table and bit layout used for both .cta_group::1 and .cta_group::2;
// .cta_group is an instruction-mnemonic qualifier, not a descriptor bit
// field, so this bit layout is independent of cta_group and only the M
// field's *value* differs from the cta_group::1 descriptor): dense
// (sparsity=0), no
// integer saturation (n/a for .kind::f16), dtype=FP32(1), atype=BF16(1),
// btype=BF16(1), no negate, no transpose, N = encoded N>>3, M = encoded
// kMGlobal>>4 = 256>>4 = 16 (the joint M across the CTA pair -- PTX ISA 9.3
// Table 44's cta_group::2/.kind::f16/Dense row lists 128xNxK/256xNxK with
// N={16,32,...,256} step 16, K=16, confirming M=256 is a valid encoded shape
// for this exact kind/cta_group/sparsity combination), no .ws B-matrix reuse
// shift. Every reserved bit is left at 0.
// ---------------------------------------------------------------------------
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

// validate_instruction_descriptor: field-by-field re-derivation of
// make_instruction_descriptor's output, independent of that function's own
// shift/mask expressions, so a regression in either one fails to compile.
// Re-checked for M=256 specifically against the official M=256 data path
// rather than assumed from the M=128 case.
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

// ---------------------------------------------------------------------------
// tcgen05 inline-PTX primitives, cta_group::2 only. CUDA 13.1's <cuda/ptx>
// header does not wrap the tcgen05 family
// (Blackwell-only), so these are hand-written from the syntax in PTX ISA 9.3
// sections 9.7.17.7 (alloc/dealloc/relinquish), 9.7.17.10.9.1 (mma), and
// 9.7.17.12.1 (commit, including the .multicast::cluster form). Every one of
// these was compiled in isolation against the pinned CUDA 13.1.80 toolchain
// and its real SASS lowering inspected before being assembled here. The generic
// mbarrier/fence/elect/cluster-rank/cluster-barrier operations reuse
// cuda::ptx:: wrappers confirmed present in this pinned toolchain's
// <cuda/ptx> header.
// ---------------------------------------------------------------------------

// One tcgen05.mma.cta_group::2.kind::f16 issue. enable_input_d is passed as
// a plain 0/1 register and converted to a PTX predicate inside the asm
// block, matching PTX ISA 9.3 section 9.7.17.10.9.1's syntax form 1 (SS
// dense, no block scaling): "tcgen05.mma.cta_group.kind [d-tmem], a-desc,
// b-desc, idesc, {disable-output-lane}, enable-input-d;". disable-output-
// lane is optional and omitted here, matching the ISA's own worked example
// in section 9.7.17.10.9.1 ("tcgen05.mma.cta_group::2.kind::tf32 [taddr0],
// adesc, bdesc, idesc, p;", section 9.7.17.12.1 Example 2) and the
// equivalent cta_group::1 omission.
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

// tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.
// multicast::cluster.b64 [mbar], ctaMask -- tracks completion of every
// asynchronous tcgen05.mma issued so far by this thread (PTX ISA 9.3 section
// 9.7.17.12.1) and multicasts the arrive-on signal to the mbarrier objects
// of every CTA selected by ctaMask, at the SAME relative shared-memory
// offset as mbar in each destination CTA (ISA text, section 9.7.17.12.1:
// "The mbarrier signal is multicast to the same offset as mbar in the
// shared memory of each destination CTA"). Only CTA rank 0's elected leader
// thread ever calls this (step 8 below); ctaMask is the
// frozen 0x0003 (bits 0 and 1: cluster_ctarank 0 and 1, i.e. both CTAs of
// this exactly-two-CTA cluster).
__device__ __forceinline__ void commit_umma_2sm_multicast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "h"(cta_mask)
        : "memory");
}

// tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32: issued
// collectively by every thread of warp 0 IN BOTH CTAs of the pair (Issue
// Granularity, PTX ISA 9.3 Table 51: "Issue from two warps, one in each of
// the current CTA and its Peer CTA, in order to collectively perform the
// operation"). dst_smem_addr is a shared-memory address, in the CALLING
// thread's own CTA, that receives that CTA's allocated Tensor Memory
// address (each CTA of the pair has its own physical Tensor Memory).
__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t dst_smem_addr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(dst_smem_addr), "r"(n_cols)
                 : "memory");
}

// tcgen05.dealloc.cta_group::2: collectively by warp 0 of both CTAs (same
// Issue Granularity entry as alloc). PTX ISA 9.3 section 9.7.17.7.1: "If
// .cta_group::2 is specified, issuing warp and peer CTA warp must
// synchronize Tensor Memory accesses before attempting to collectively
// deallocate the Tensor Memory, and tcgen05.dealloc may block to
// collectively perform the deallocation with the other peer CTA's warp" --
// hence the cluster synchronization required immediately before this call
// (step 12 below).
__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" : : "r"(taddr), "r"(n_cols) : "memory");
}

// tcgen05.relinquish_alloc_permit.cta_group::2: collectively by warp 0 of
// both CTAs (same Issue Granularity entry as alloc/dealloc).
__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" : : : "memory");
}

// tcgen05.fence::after_thread_sync and tcgen05.ld/tcgen05.wait::ld take no
// .cta_group qualifier at all (PTX ISA 9.3 Table 51 lists cta_group as N/A
// for .ld/.st/.wait, and section 9.7.17.11.1 shows .fence with no cta_group
// qualifier either): every thread accesses only the Tensor Memory of its own
// executing CTA regardless of whether the preceding MMA was cta_group::1 or
// cta_group::2. These are therefore identical in form to the cta_group::1
// equivalents, re-derived here rather than shared.
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

// ---------------------------------------------------------------------------
// TMEM load address construction (PTX ISA 9.3 section 9.7.17.1.1, "Tensor
// Memory Addressing"; section 9.7.17.8.1, "Access restrictions"; and section
// 9.7.17.10.5.1's Layout A / Figure 205-206 for the M=256 data path): each
// CTA's own Tensor Memory has 128 lanes, split into four 32-lane chunks (one
// per warp of the warpgroup: warp 0 -> lanes 0-31, warp 1 -> 32-63, warp 2
// -> 64-95, warp 3 -> 96-127). Figure 205/206 decompose the joint M=256
// output as (warp-rank%4, even/odd CTA in the CTA pair): each CTA's own
// local M-rows 0-127 (warp-rank 0-3) map to that SAME CTA's own local TMEM
// lanes 0-127 -- there is no separate "TMEM lanes 128-255" to address,
// because Tensor Memory is per-CTA. This mapping is therefore IDENTICAL in
// form to the single-CTA addressing; only the *global* row index (computed
// separately, at the point where D is written to global memory) differs by
// an additive cta_rank*128 term.
// ---------------------------------------------------------------------------
constexpr uint32_t kTmemLaneShift = 16;        // PTX ISA 9.3 9.7.17.1.1: lane index occupies bits 31-16.
constexpr uint32_t kTmemRowsPerWarp = 32;      // PTX ISA 9.3 9.7.17.8.1: each warp owns a 32-lane chunk.
constexpr uint32_t kTmemColsPerFragment = 32;  // This kernel's fixed 32-column tcgen05.ld.x32 fragment width.

__device__ __forceinline__ uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {
    const uint32_t lane_contribution = (static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift;
    const uint32_t column_contribution = static_cast<uint32_t>(frag) * kTmemColsPerFragment;
    return tmem_base + lane_contribution + column_contribution;
}

// ---------------------------------------------------------------------------
// mbarrier-initialization fence: fence.mbarrier_init.release.cluster (PTX
// ISA 9.3, "Parallel Synchronization and Communication Instructions ->
// Membar/Fence Instructions"). This is a DIFFERENT fence from
// fence.proxy.async (issued separately, immediately afterward -- see step
// 3-4 in umma_2sm_body below) and neither substitutes for the other:
// fence.proxy.async publishes ordinary (generic-proxy) memory writes -- here,
// the A/B shared-memory fill from step 1 -- to the ASYNC proxy that
// tcgen05.mma reads through (PTX ISA 9.3 section 9.7.17.6.5). This fence
// instead publishes the INITIALIZATION performed by mbarrier.init itself to
// every thread of the cluster, so that a later arrive-on operation targeting
// this mbarrier from the PEER CTA (CTA rank 0's multicast
// tcgen05.commit...multicast::cluster, which arrives on CTA rank 1's own
// local mbarrier at the identical relative SMEM offset) is guaranteed to
// observe a fully initialized barrier object
// rather than racing its initialization.
//
// The pinned CUDA 13.1.80 toolchain's <cuda/ptx> header wraps this exact
// instruction (cuda/__ptx/instructions/generated/fence_mbarrier_init.h,
// included transitively by the top-level <cuda/ptx> header via
// cuda/__ptx/instructions/fence.h, confirmed present in this pinned image
// and unconditionally lowering to "fence.mbarrier_init.release.cluster;" for
// __CUDA_ARCH__ >= 900, which sm_103a satisfies), so this helper uses the
// official wrapper rather than hand-written inline PTX -- matching this
// file's existing convention for every other primitive with genuine
// <cuda/ptx> coverage (mbarrier_init, fence_proxy_async, elect_sync,
// get_sreg_cluster_ctarank/nctarank, barrier_cluster_arrive/wait,
// mbarrier_try_wait_parity). Only the tcgen05 family (unwrapped by this
// pinned toolchain, Blackwell-only) uses hand-written inline PTX in this
// file.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
}

// ---------------------------------------------------------------------------
// Launch contract: every visible kernel must reject any launch that is not
// exactly grid=(2,1,1), cluster=(2,1,1), block=(128,1,1). __cluster_dims__
// and __launch_bounds__ only constrain the compiled kernel's launch
// requirements; this is an explicit device-side guard, evaluated before any
// __syncthreads(), cluster barrier, mbarrier initialization, TMEM
// allocation, or UMMA instruction (see the top of umma_2sm_body below), so a
// rejected launch can never leave TMEM allocated or block on a barrier. The
// predicate depends only on values that are uniform across the whole
// cluster (gridDim, blockDim, %cluster_nctarank) plus the ISA-guaranteed
// range fact "0 <= %cluster_ctarank < %cluster_nctarank" (PTX ISA 9.3
// section 10.16), so both CTAs of the pair independently compute the SAME
// accept/reject verdict: no accepted launch ever lets one CTA proceed into a
// collective operation while its peer has rejected and returned.
// ---------------------------------------------------------------------------
constexpr int kExpectedGridDim = kGridBlocks;
constexpr int kExpectedBlockDimX = kThreadsPerCta;
constexpr uint32_t kExpectedClusterCtas = kClusterCtas;

__device__ __forceinline__ bool launch_contract_is_valid(uint32_t cluster_nctarank, uint32_t cluster_ctarank) {
    return gridDim.x == kExpectedGridDim && gridDim.y == 1 && gridDim.z == 1 &&
           blockDim.x == kExpectedBlockDimX && blockDim.y == 1 && blockDim.z == 1 &&
           cluster_nctarank == kExpectedClusterCtas && cluster_ctarank < kExpectedClusterCtas;
}

// ---------------------------------------------------------------------------
// Device: templated kernel body, instantiated once per (N, DEPTH)
// specialization by the extern "C" wrappers below. g_d_out holds the joint
// 256 x N FP32 output (both CTAs write into the same buffer, at disjoint row
// ranges); g_elapsed_cycles is written only by CTA rank 0's elected leader;
// g_launch_ok[0]/g_launch_ok[1] are written by CTA rank 0/1 respectively so
// the host can observe both ranks' launch-contract outcome independently.
// ---------------------------------------------------------------------------
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

    // ---- Launch contract: must precede any __syncthreads(), cluster ------
    // ---- barrier, mbarrier init, TMEM allocation, or UMMA instruction. ---
    // ---- A rejected launch writes 0 to this CTA's own slot and returns ---
    // ---- immediately, before touching any shared or cluster state, so ---
    // ---- it can never allocate TMEM or block on a barrier. Both ranks' ---
    // ---- slots let the host confirm the rejection (or acceptance) was ---
    // ---- uniform across the pair. -----------------------------------
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

    // ---- Step 1: fill this CTA's local A (128 rows) and B (N/2 cols) ----
    // ---- the frozen validation pattern, placed into the fixed K-major ----
    // ---- physical layout. A's *value* depends on the GLOBAL row (cta_rank
    // ---- * 128 + local_row) so the two CTA halves cannot be accidentally -
    // ---- exchanged or duplicated; its *physical* SMEM position ---------
    // ---- stays local (128 rows) since Tensor Memory and Shared ----------
    // ---- Memory are both per-CTA. The cta_group::2 B layout likewise -----
    // ---- assigns N/2 logical columns to each peer CTA: CTA rank 0 stores -
    // ---- global columns [0,N/2), CTA rank 1 stores [N/2,N), both at ------
    // ---- local SMEM column positions [0,N/2). -----------------------------
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

    // ---- Step 2: local synchronization after initialization. -------------
    __syncthreads();

    // ---- Step 3-4: one local mbarrier per CTA (same relative SMEM offset -
    // ---- in both CTAs by construction: both CTAs execute this identical --
    // ---- kernel body, so the compiler lays out __shared__ variables ------
    // ---- identically for every CTA instance), plus TWO required, ---------
    // ---- DIFFERENT fences (see fence_mbarrier_init_release_cluster's own -
    // ---- comment above for why one does not substitute for the other): ---
    // ---- first fence.mbarrier_init.release.cluster, publishing THIS ------
    // ---- mbarrier's own initialization to the cluster so the peer CTA's --
    // ---- later arrive-on operation cannot race it; then fence.proxy.async
    // ---- at CLUSTER scope (space_cluster -> fence.proxy.async.shared:: ---
    // ---- cluster), not merely CTA scope: this CTA's own A/B writes must --
    // ---- become visible to the OTHER CTA's hardware, since the joint -----
    // ---- cta_group::2 MMA (issued by CTA rank 0 only) reads CTA rank 1's -
    // ---- own local A at the identical relative offset. A cluster-scoped --
    // ---- fence is a strict superset of a CTA-scoped one, so it is the ----
    // ---- safe choice under either reading of the CTA-pair hardware -------
    // ---- mechanism, and it costs nothing here since it runs outside ------
    // ---- the timed region. -----------------------------------------------
    // ---- fence.proxy.async does NOT itself publish the mbarrier's own ----
    // ---- initialization (it fences the async proxy's view of GENERIC- ----
    // ---- proxy memory writes -- the A/B fill from step 1 -- not the ------
    // ---- mbarrier object's own init flag), so neither fence may be -------
    // ---- dropped or merged into the other. --------------------------------
    if (tid == 0) {
        cuda::ptx::mbarrier_init(&mbar, 1u);
        fence_mbarrier_init_release_cluster();
        cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);
    }
    // Publish the mbarrier initialization and both fences' effects to every
    // thread, including whichever thread elect_sync selects as leader below;
    // the cluster rendezvous immediately below (step 5) is what publishes
    // them on to the peer CTA before any collective TMEM operation.
    __syncthreads();

    bool is_leader = false;
    if (tid < 32) {
        is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    }

    // ---- Step 5: cluster synchronization before collective TMEM ops. -----
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    // ---- Step 6: tcgen05.alloc.cta_group::2 issued collectively by -------
    // ---- warp 0 of BOTH CTAs (never a single elected lane, never only ----
    // ---- rank 0): allocation, deallocation, and relinquishing are --------
    // ---- warp-collective operations. Exactly N columns. ------------------
    if (warp_id == 0) {
        const uint32_t tmem_addr_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_shared));
        tcgen05_alloc_2sm(tmem_addr_smem, static_cast<uint32_t>(N));
    }
    __syncthreads();
    const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr_shared);

    // ---- Step 7: cluster synchronization before using the allocated ------
    // ---- TMEM (both CTAs must have their own tmem_d published locally ----
    // ---- and be certain the peer has too, before rank 0's leader issues --
    // ---- the joint MMA). --------------------------------------------------
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    const uint64_t a_desc = make_smem_descriptor(A);
    const uint64_t b_desc = make_smem_descriptor(B);
    constexpr uint32_t idesc = make_instruction_descriptor<N>();
    const uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    // ---- Steps 8-10, timed region: only CTA rank 0's elected leader ------
    // ---- issues the depth-unrolled UMMA burst and its multicast commit ---
    // ---- of the burst: only one elected thread in CTA rank 0 may issue ---
    // ---- the MMA burst and its commit. BOTH CTAs' elected leaders wait ---
    // ---- on their OWN local mbarrier every iteration; that wait is never -
    // ---- enclosed in a cta_rank == 0 condition. Every outer iteration ----
    // ---- restarts with enable-input-d=false, so D is not accumulated -----
    // ---- across separate outer iterations, matching the 1-SM arm's -------
    // ---- depth-burst semantic. Only CTA rank 0's leader records ----------
    // ---- %clock64, and only when timing_mode == kTimed. ------------------
    //
    // ---- Required per-phase CTA-pair handshake: PTX requires at least one
    // ---- successful test_wait/try_wait observation of a given mbarrier ---
    // ---- phase before any later arrive-on operation (here, the NEXT ------
    // ---- iteration's multicast commit) targets that same mbarrier again. -
    // ---- CTA rank 0's leader must therefore not be allowed to start phase-
    // ---- P+1's commit until it is certain CTA rank 1's leader has already-
    // ---- SUCCESSFULLY observed phase P's completion -- not merely that ---
    // ---- rank 0 issued it. Each CTA's own successful local wait is first -
    // ---- published to that whole CTA with __syncthreads(), and then EVERY
    // ---- thread of BOTH CTAs (never just leaders, never just rank 0) -----
    // ---- rendezvouses with a full cluster arrive/wait before the loop can-
    // ---- take its back-edge and rank 0 can issue another commit. This ----
    // ---- sequence therefore runs once per mbarrier phase (i.e. inside ----
    // ---- every outer iteration, before the next commit), not once after -
    // ---- the whole loop -- and, because it sits between the two timed ----
    // ---- clock reads below, its cost is included in the measured region. -
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

            // Each CTA's own elected leader waits on that CTA's own local
            // mbarrier; never enclosed in a cta_rank == 0 condition (CTA
            // rank 1's leader must wait too, even though it never issues
            // anything).
            while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {
            }
            parity ^= 1u;
        }

        // Publish this CTA's successful local wait to every thread of this
        // CTA (the other 127 threads have no wait of their own to publish,
        // but must still rendezvous here so the cluster barrier below is
        // reached uniformly by the whole CTA).
        __syncthreads();

        // Every non-exited thread in BOTH CTAs participates here -- not
        // just leaders, not just rank 0. Rank 0 cannot start the next
        // iteration's commit until this cluster barrier releases, which
        // cannot happen until CTA rank 1's leader has also reached it,
        // which in turn cannot happen until CTA rank 1's leader has itself
        // successfully observed this phase's completion above.
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

    // ---- Step 11: untimed TMEM readback, executed by BOTH CTAs, every ----
    // ---- thread (never enclosed in a cta_rank == 0 condition). Each CTA --
    // ---- reads only its own local 128 TMEM rows; CTA rank 1 does NOT -----
    // ---- add 128 to its local TMEM lane address (Tensor Memory is per- ---
    // ---- CTA). The rank offset belongs ----------------------------------
    // ---- only in the GLOBAL output index below. ------------------------
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

    // ---- Step 12: cluster synchronization after the final TMEM access. ---
    __syncthreads();
    cuda::ptx::barrier_cluster_arrive();
    cuda::ptx::barrier_cluster_wait();

    // ---- Steps 13-14: dealloc and relinquish, collectively by warp 0 of --
    // ---- BOTH CTAs (never a single elected lane, never only rank 0). -----
    if (warp_id == 0) {
        tcgen05_dealloc_2sm(tmem_d, static_cast<uint32_t>(N));
        tcgen05_relinquish_alloc_permit_2sm();
    }

    // ---- Step 15: final mbarrier invalidation, after it can no longer ----
    // ---- be referenced by any pending collective operation. --------------
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];"
                     :
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar)))
                     : "memory");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The twelve extern "C" specializations. Every
// kernel uses __cluster_dims__(2, 1, 1) __launch_bounds__(128); the only
// valid launch geometry is grid=(2,1,1), cluster=(2,1,1), block=(128,1,1).
// ---------------------------------------------------------------------------
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

// ---- CPU reference: logical formulas only, independent ------------------
// ---- of the GPU kernel's physical SMEM/TMEM packing. Uses the GLOBAL ------
// ---- row (0..255) exactly as the device-side A initialization does, so ---
// ---- a wrong global-row mapping, rank duplication, or missing rank -------
// ---- offset on the device side is caught as a numerical mismatch. --------
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

// Validates all kMGlobal (256) * n elements -- both CTA ranks' output rows,
// not only rank 0's.
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

// ---------------------------------------------------------------------------
// GPU environment/provenance: exactly one visible device, logical device 0,
// compute capability 10.3, and its CUDA-reported UUID (never nvidia-smi)
// matches EXPECTED_GPU_UUID.
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
// CLI. Same flag surface as the 1-SM arm's host interface.
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
        "umma_2sm - 2-SM BF16 UMMA (tcgen05.mma, kind::f16, cta_group::2) instruction microbenchmark\n"
        "\n"
        "2-SM CTA-pair arm of the BF16 UMMA throughput experiment. One static\n"
        "two-CTA cluster; CTA rank 0's elected leader issues depth-many\n"
        "tcgen05.mma.cta_group::2 instructions per iteration and a cluster-multicast\n"
        "commit; both CTAs wait, read back, and report elapsed cycles. Not a\n"
        "TFLOP/s or saturation claim: this measures instruction issue, not\n"
        "sustained tensor-core throughput.\n"
        "\n"
        "Usage:\n"
        "  umma_2sm --help\n"
        "  umma_2sm --self-test\n"
        "  umma_2sm --run-kind {smoke,benchmark} --n {64,128,256} --depth {4,16,64,256} \\\n"
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
// CSV: same schema/column order as the 1-SM arm's CSV, with this arm's
// frozen fixed values (method=umma_2sm, cta_group=2, m=256,
// threads_per_cta=128, grid_blocks=2).
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
        << kMethodName << ',' << r.sample_index << ',' << kCtaGroup << ',' << kMGlobal << ',' << r.n << ','
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
// validation, shared by the main run path and --self-test. The launch
// geometry is fixed at grid=(2,1,1), block=(128,1,1); __cluster_dims__(2,1,1)
// on the kernel itself fixes the cluster shape at compile time, so no
// separate cluster-aware launch API is needed (CUDA C++ Programming Guide,
// "Thread Block Clusters": a compile-time cluster size is used directly with
// the ordinary <<<grid, block, smem>>> launch syntax).
// ---------------------------------------------------------------------------
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
    // 0 means "not confirmed": both an explicit device-side rejection and a
    // kernel that never ran at all collapse to this same not-OK value.
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

    // The host launcher below always requests grid=(2,1,1) cluster=(2,1,1)
    // block=(128,1,1), so this can only fire if the launch-contract check
    // itself regresses; it must still abort loudly rather than silently
    // validating garbage D. Both ranks are required to confirm, uniformly.
    if (launch_ok_host[0] != 1 || launch_ok_host[1] != 1) {
        fail("launch-contract violation for %s: kernel did not confirm grid=(2,1,1) "
             "cluster=(2,1,1) block=(128,1,1) (launch_ok=[%d,%d])",
             spec.symbol, launch_ok_host[0], launch_ok_host[1]);
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
                 "umma_2sm: ERROR: correctness validation FAILED for %s: mismatches=%lld "
                 "first_index=%lld expected=%.1f obtained=%.1f max_abs_error=%.6g\n",
                 spec.symbol, (long long)validation.mismatches, (long long)validation.first_mismatch_index,
                 validation.first_mismatch_expected, validation.first_mismatch_obtained,
                 validation.max_abs_error);
    fail("correctness validation failed for %s; no timing or CSV output was produced", spec.symbol);
}

// Runs one specialization untimed (timing_mode=kUntimed, so neither %clock64
// read executes) and aborts on any correctness mismatch. Used for
// pre-timing validation and warm-up, neither of which may be timed.
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
    std::fprintf(stderr, "umma_2sm: SELF_TEST start\n");
    int passed = 0;
    for (const auto& spec : kSpecializations) {
        const RunResult result = run_once(spec, kSelfTestIterations, TimingMode::kUntimed);
        std::fprintf(stderr,
                     "umma_2sm: SELF_TEST %s n=%d depth=%d result=%s mismatches=%lld "
                     "max_abs_error=%.6g\n",
                     spec.symbol, spec.n, spec.depth, result.validation.ok ? "PASS" : "FAIL",
                     (long long)result.validation.mismatches, result.validation.max_abs_error);
        if (result.validation.ok) ++passed;
    }
    std::fprintf(stderr, "umma_2sm: SELF_TEST_RESULT %d/%d\n", passed, 12);
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
        std::fprintf(stderr, "umma_2sm: ERROR: %s\n", parse_err.c_str());
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
                 "umma_2sm: run_kind=%s n=%d depth=%d iterations=%lld warmup_iterations=%lld "
                 "repetitions=%lld\n",
                 cli.run_kind.c_str(), cli.n, cli.depth, (long long)cli.iterations,
                 (long long)cli.warmup_iterations, (long long)cli.repetitions);

    // Step 1: validate without timing before anything else.
    run_untimed_or_die(*spec, cli.iterations);
    std::fprintf(stderr, "umma_2sm: correctness=OK mismatches=0 (pre-timing check)\n");

    // Step 2: warm-up, discarded, also untimed.
    for (int64_t w = 0; w < cli.warmup_iterations; ++w) {
        run_untimed_or_die(*spec, cli.iterations);
    }

    // Steps 3-5: timed repetitions, each re-validated before its CSV row.
    print_csv_header();
    const int64_t flops_per_umma =
        checked_mul_i64(checked_mul_i64(2, kMGlobal, "2*M"), checked_mul_i64(spec->n, kK, "N*K"), "flops_per_umma");
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
        std::fprintf(stderr, "umma_2sm: ERROR: %d resource cleanup failure(s) occurred\n",
                     g_cleanup_failures);
        return 1;
    }
    return 0;
}
