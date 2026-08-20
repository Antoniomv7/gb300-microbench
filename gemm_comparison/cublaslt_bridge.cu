// P3.5 - minimal C-ABI bridge to cuBLASLt for the five frozen BF16 GEMM shapes.
//
// This file owns no GEMM kernel. It creates a cuBLASLt handle, the operation
// descriptor, four matrix layouts, a preference object, queries the vendor
// heuristic, validates the selected algorithm, allocates exactly the workspace
// that algorithm requires, exports the selected algorithm's metadata, and
// launches `cublasLtMatmul`. Nothing else. No NVIDIA source is copied, forked,
// patched, or vendored: only the public cuBLASLt API declared in the pinned
// CUDA 13.1 headers is called.
//
// Difference from the closed P3.3 bridge (`cublaslt_bridge.cu`, untouched):
// P3.3 froze one single shape as compile-time constants, so `p33_plan_create`
// took no geometry at all. P3.5 must serve five shapes, so `p35_plan_create`
// accepts (M,N,K) - and then refuses every geometry that is not one of the five
// entries in the frozen allowlist below. Nothing else about the descriptor
// contract or the algorithm policy is parameterised, negotiated, or reachable
// from the caller: the transpose modes, memory orders, data types, compute and
// scale types, pointer mode, epilogue, alpha, beta, workspace limit, requested
// heuristic count, and search mode are all still compile-time constants, and
// the leading dimensions are *derived* from the validated shape rather than
// supplied. Every one of them is reported back through `p35_plan_info_t`, so
// the Python wrapper can assert that this translation unit and the wrapper
// agree, instead of restating the contract twice and hoping.
//
// The allowlist is also readable from the caller (`p35_shape_count` /
// `p35_shape_at`), which lets the Python wrapper prove that the C side's five
// shapes are exactly the five shapes it froze independently - rather than the
// two sides silently disagreeing about what "the frozen shapes" means.
//
// Contract with the caller:
//
//   * Every exported function is `extern "C"` and returns `int` (0 on success,
//     non-zero on failure) or a plain pointer/size. No C++ exception may cross
//     the boundary: every entry point has a catch-all handler.
//   * Nothing is ever written to stdout or stderr. A failure records a message
//     retrievable with `p35_last_error()`.
//   * The bridge never chooses a fallback. If a frozen configuration has no
//     supported algorithm, it fails and says so.
//
// Algorithm policy (fixed, non-autotuned, never benchmarked; identical to the
// closed P3.3 policy):
//
//   * workspace limit exactly P35_WORKSPACE_LIMIT_BYTES;
//   * exactly P35_HEURISTIC_REQUESTED heuristic results requested;
//   * CUBLASLT_SEARCH_BEST_FIT;
//   * the first returned entry whose state is CUBLAS_STATUS_SUCCESS is taken;
//   * that algorithm is re-validated with cublasLtMatmulAlgoCheck();
//   * it is rejected if it needs more workspace than the fixed limit;
//   * exactly the required workspace is allocated (a null pointer when the
//     requirement is zero);
//   * only that one algorithm is ever executed.
//
// A different supported algorithm may naturally be selected for each shape.
// The *selection policy* never changes.
//
// No candidate is timed, compared, or ranked here: this bridge measures
// nothing. The Python wrapper owns every timer.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

// --- Frozen configuration ----------------------------------------------------
//
// C = A x B^T, BF16 x BF16 -> FP32 with FP32 accumulation. A is M x K row-major
// (lda = K), B is N x K row-major (ldb = K), C and D are M x N row-major
// (ldc = ldd = N). These match the physical layouts the pinned CuTe DSL tensor
// factory produces for a_major=k, b_major=k, c_major=n, so no data is
// transposed, copied, or reinterpreted anywhere in this unit.

#define P35_ABI_VERSION 1

// The five and only five (M,N,K) shapes, in the frozen P3.5 order. L is always
// 1. This allowlist is the C-side half of the shape contract; the Python
// wrapper freezes the same five independently and the two are compared.
static const int64_t P35_SHAPES[][3] = {
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {16384, 512, 4096},
    {32768, 512, 4096},
    {512, 16384, 4096},
};
static const size_t P35_SHAPE_COUNT = sizeof(P35_SHAPES) / sizeof(P35_SHAPES[0]);

static const int64_t P35_BATCH_COUNT = 1;

// A conservative independent ceiling on any single extent. Every frozen extent
// is far below it; it exists so the overflow arithmetic below has a proven
// bound even if the allowlist were ever edited.
static const int64_t P35_MAX_EXTENT = 1073741824;  // 2^30

static const cublasOperation_t P35_TRANSA = CUBLAS_OP_N;
static const cublasOperation_t P35_TRANSB = CUBLAS_OP_T;

static const cublasLtOrder_t P35_ORDER = CUBLASLT_ORDER_ROW;

static const cudaDataType_t P35_AB_TYPE = CUDA_R_16BF;
static const cudaDataType_t P35_CD_TYPE = CUDA_R_32F;
static const cublasComputeType_t P35_COMPUTE_TYPE = CUBLAS_COMPUTE_32F;
static const cudaDataType_t P35_SCALE_TYPE = CUDA_R_32F;

// Element widths, used only by the overflow proof below.
static const int64_t P35_AB_ELEMENT_BYTES = 2;
static const int64_t P35_CD_ELEMENT_BYTES = 4;

static const cublasLtPointerMode_t P35_POINTER_MODE = CUBLASLT_POINTER_MODE_HOST;
static const cublasLtEpilogue_t P35_EPILOGUE = CUBLASLT_EPILOGUE_DEFAULT;

static const float P35_ALPHA = 1.0f;
static const float P35_BETA = 0.0f;

// Exactly 64 MiB. Never negotiated, never retried with another value.
static const uint64_t P35_WORKSPACE_LIMIT_BYTES = 67108864ULL;

// Exactly 32 heuristic results are requested; the first supported one wins.
static const int P35_HEURISTIC_REQUESTED = 32;

static const cublasLtMatmulSearch_t P35_SEARCH_MODE = CUBLASLT_SEARCH_BEST_FIT;

// The largest pointer alignment this bridge will ever claim. cuBLASLt's own
// default for the minimum-alignment preferences is 256 bytes; claiming more
// than a pointer actually satisfies would let the heuristic return an
// algorithm the data cannot legally feed.
static const uint32_t P35_MAX_ALIGNMENT_BYTES = 256u;

// --- Error reporting ---------------------------------------------------------

static thread_local char g_last_error[1024] = {0};

static void p35_clear_error(void) { g_last_error[0] = '\0'; }

static void p35_set_error(const char* message) {
    if (message == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

static void p35_set_error_status(const char* what, cublasStatus_t status) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s failed with cublasStatus_t=%d",
                  what, static_cast<int>(status));
}

static void p35_set_error_cuda(const char* what, cudaError_t status) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s failed with cudaError_t=%d (%s)",
                  what, static_cast<int>(status), cudaGetErrorString(status));
}

// --- Exported metadata -------------------------------------------------------
//
// Every member is 8 bytes wide so the struct has one unambiguous layout with
// no padding, which keeps the ctypes mirror on the Python side exact. The
// wrapper additionally checks sizeof() against p35_plan_info_size().

extern "C" {

typedef struct P35PlanInfo {
    int64_t abi_version;
    int64_t cublaslt_version;

    int64_t shape_index;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t batch_count;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    int64_t transa;
    int64_t transb;
    int64_t order_a;
    int64_t order_b;
    int64_t order_c;
    int64_t order_d;
    int64_t type_a;
    int64_t type_b;
    int64_t type_c;
    int64_t type_d;
    int64_t compute_type;
    int64_t scale_type;
    int64_t pointer_mode;
    int64_t epilogue;

    int64_t search_mode;
    int64_t workspace_limit_bytes;
    int64_t workspace_bytes;
    int64_t workspace_is_null;

    int64_t heuristic_requested;
    int64_t heuristic_returned;
    int64_t heuristic_index;

    int64_t alignment_a_bytes;
    int64_t alignment_b_bytes;
    int64_t alignment_c_bytes;
    int64_t alignment_d_bytes;

    int64_t algo_id;
    int64_t tile_id;
    int64_t stages_id;
    int64_t split_k;
    int64_t reduction_scheme;
    int64_t cta_swizzling;
    int64_t custom_option;
    int64_t inner_shape_id;
    int64_t cluster_shape_id;

    double waves_count;
    double alpha;
    double beta;
} P35PlanInfo;

}  // extern "C"

// --- Plan --------------------------------------------------------------------

struct P35Plan {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t layout_a;
    cublasLtMatrixLayout_t layout_b;
    cublasLtMatrixLayout_t layout_c;
    cublasLtMatrixLayout_t layout_d;
    cublasLtMatmulAlgo_t algo;

    const void* a;
    const void* b;
    const void* c;
    void* d;

    void* workspace;
    size_t workspace_bytes;

    cudaStream_t stream;
};

static void p35_note_cublaslt_cleanup_failure(const char* what,
                                              cublasStatus_t status,
                                              bool preserve_original_error,
                                              int* failed) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    if (*failed == 0 && !preserve_original_error) {
        p35_set_error_status(what, status);
    }
    *failed = 1;
}

static void p35_note_cuda_cleanup_failure(const char* what,
                                          cudaError_t status,
                                          bool preserve_original_error,
                                          int* failed) {
    if (status == cudaSuccess) {
        return;
    }
    if (*failed == 0 && !preserve_original_error) {
        p35_set_error_cuda(what, status);
    }
    *failed = 1;
}

// Attempt every release even after one fails. When cleanup follows a primary
// creation or execution error, retain that diagnostic; when cleanup is the
// only failure, record the first release error and return non-zero.
static int p35_plan_release(P35Plan* plan, bool preserve_existing_error) {
    if (plan == nullptr) {
        return 0;
    }

    char original_error[sizeof(g_last_error)] = {0};
    const bool preserve_original_error =
        preserve_existing_error && g_last_error[0] != '\0';
    if (preserve_original_error) {
        std::snprintf(original_error, sizeof(original_error), "%s", g_last_error);
    }

    int failed = 0;
    if (plan->workspace != nullptr) {
        const cudaError_t status = cudaFree(plan->workspace);
        p35_note_cuda_cleanup_failure(
            "cudaFree(cuBLASLt workspace)", status, preserve_original_error, &failed);
        plan->workspace = nullptr;
    }
    if (plan->layout_d != nullptr) {
        const cublasStatus_t status = cublasLtMatrixLayoutDestroy(plan->layout_d);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtMatrixLayoutDestroy(D)", status, preserve_original_error, &failed);
        plan->layout_d = nullptr;
    }
    if (plan->layout_c != nullptr) {
        const cublasStatus_t status = cublasLtMatrixLayoutDestroy(plan->layout_c);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtMatrixLayoutDestroy(C)", status, preserve_original_error, &failed);
        plan->layout_c = nullptr;
    }
    if (plan->layout_b != nullptr) {
        const cublasStatus_t status = cublasLtMatrixLayoutDestroy(plan->layout_b);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtMatrixLayoutDestroy(B)", status, preserve_original_error, &failed);
        plan->layout_b = nullptr;
    }
    if (plan->layout_a != nullptr) {
        const cublasStatus_t status = cublasLtMatrixLayoutDestroy(plan->layout_a);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtMatrixLayoutDestroy(A)", status, preserve_original_error, &failed);
        plan->layout_a = nullptr;
    }
    if (plan->operation != nullptr) {
        const cublasStatus_t status = cublasLtMatmulDescDestroy(plan->operation);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtMatmulDescDestroy", status, preserve_original_error, &failed);
        plan->operation = nullptr;
    }
    if (plan->handle != nullptr) {
        const cublasStatus_t status = cublasLtDestroy(plan->handle);
        p35_note_cublaslt_cleanup_failure(
            "cublasLtDestroy", status, preserve_original_error, &failed);
        plan->handle = nullptr;
    }
    delete plan;

    if (preserve_original_error) {
        p35_set_error(original_error);
    }
    return failed;
}

static int p35_preference_release(cublasLtMatmulPreference_t* preference,
                                  bool preserve_existing_error) {
    if (preference == nullptr || *preference == nullptr) {
        return 0;
    }

    char original_error[sizeof(g_last_error)] = {0};
    const bool preserve_original_error =
        preserve_existing_error && g_last_error[0] != '\0';
    if (preserve_original_error) {
        std::snprintf(original_error, sizeof(original_error), "%s", g_last_error);
    }

    const cublasLtMatmulPreference_t owned = *preference;
    *preference = nullptr;
    const cublasStatus_t status = cublasLtMatmulPreferenceDestroy(owned);
    int failed = 0;
    p35_note_cublaslt_cleanup_failure(
        "cublasLtMatmulPreferenceDestroy", status, preserve_original_error, &failed);
    if (preserve_original_error) {
        p35_set_error(original_error);
    }
    return failed;
}

// The largest power of two, capped at P35_MAX_ALIGNMENT_BYTES, that divides
// the given address. Never overstates: the returned value is a true divisor of
// the pointer, so the minimum-alignment preference derived from it can only
// exclude algorithms the buffer genuinely cannot satisfy.
static uint32_t p35_pointer_alignment(const void* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address == 0) {
        return 0u;
    }
    uint32_t alignment = 1u;
    while (alignment < P35_MAX_ALIGNMENT_BYTES &&
           (address % (static_cast<uintptr_t>(alignment) * 2u)) == 0u) {
        alignment *= 2u;
    }
    return alignment;
}

// Multiplies two positive int64 values, failing closed on overflow instead of
// wrapping. Used for every element count and byte size derived from a shape.
static int p35_mul_checked(int64_t left, int64_t right, int64_t* out) {
    if (left <= 0 || right <= 0) {
        return 1;
    }
    if (left > INT64_MAX / right) {
        return 1;
    }
    *out = left * right;
    return 0;
}

// Proves that every element count and byte size this shape implies is
// representable, before a single descriptor is created. The frozen shapes are
// all far below these bounds; the arithmetic exists so a future edit to the
// allowlist cannot silently produce a wrapped size.
static int p35_validate_extents(int64_t m, int64_t n, int64_t k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "p35_plan_create: every extent must be positive, got "
                      "M=%lld N=%lld K=%lld",
                      static_cast<long long>(m), static_cast<long long>(n),
                      static_cast<long long>(k));
        return 1;
    }
    if (m > P35_MAX_EXTENT || n > P35_MAX_EXTENT || k > P35_MAX_EXTENT) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "p35_plan_create: an extent exceeds the %lld element ceiling "
                      "(M=%lld N=%lld K=%lld)",
                      static_cast<long long>(P35_MAX_EXTENT), static_cast<long long>(m),
                      static_cast<long long>(n), static_cast<long long>(k));
        return 1;
    }

    int64_t elements = 0;
    int64_t bytes = 0;
    const struct {
        int64_t rows;
        int64_t cols;
        int64_t element_bytes;
        const char* name;
    } buffers[] = {
        {m, k, P35_AB_ELEMENT_BYTES, "A"},
        {n, k, P35_AB_ELEMENT_BYTES, "B"},
        {m, n, P35_CD_ELEMENT_BYTES, "C"},
        {m, n, P35_CD_ELEMENT_BYTES, "D"},
    };
    for (const auto& buffer : buffers) {
        if (p35_mul_checked(buffer.rows, buffer.cols, &elements) != 0 ||
            p35_mul_checked(elements, P35_BATCH_COUNT, &elements) != 0 ||
            p35_mul_checked(elements, buffer.element_bytes, &bytes) != 0) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "p35_plan_create: the size of operand %s overflows a signed "
                          "64-bit byte count",
                          buffer.name);
            return 1;
        }
        if (static_cast<uint64_t>(bytes) > static_cast<uint64_t>(SIZE_MAX)) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "p35_plan_create: the size of operand %s does not fit in size_t",
                          buffer.name);
            return 1;
        }
    }
    return 0;
}

// Returns the index of (m,n,k) in the frozen allowlist, or -1. This is the only
// gate through which a geometry reaches a descriptor: an arbitrary shape can
// never be planned, executed, or measured, whatever the caller passes.
static int p35_shape_index_of(int64_t m, int64_t n, int64_t k) {
    for (size_t index = 0; index < P35_SHAPE_COUNT; ++index) {
        if (P35_SHAPES[index][0] == m && P35_SHAPES[index][1] == n &&
            P35_SHAPES[index][2] == k) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

static int p35_create_layout(cublasLtMatrixLayout_t* layout,
                             cudaDataType_t type,
                             uint64_t rows,
                             uint64_t cols,
                             int64_t ld,
                             const char* name) {
    cublasStatus_t status = cublasLtMatrixLayoutCreate(layout, type, rows, cols, ld);
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutCreate(%s)", name);
        p35_set_error_status(what, status);
        return 1;
    }

    const int32_t order = static_cast<int32_t>(P35_ORDER);
    status = cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                              sizeof(order));
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutSetAttribute(ORDER, %s)", name);
        p35_set_error_status(what, status);
        return 1;
    }

    const int32_t batch_count = static_cast<int32_t>(P35_BATCH_COUNT);
    status = cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutSetAttribute(BATCH_COUNT, %s)",
                      name);
        p35_set_error_status(what, status);
        return 1;
    }
    return 0;
}

// Reads one algorithm configuration attribute into an int64_t, failing closed
// if the attribute is unavailable or the library writes an unexpected width.
template <typename T>
static int p35_read_algo_config(const cublasLtMatmulAlgo_t* algo,
                                cublasLtMatmulAlgoConfigAttributes_t attribute,
                                const char* name,
                                int64_t* out) {
    T value = T();
    size_t written = 0;
    const cublasStatus_t status =
        cublasLtMatmulAlgoConfigGetAttribute(algo, attribute, &value, sizeof(value), &written);
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[160];
        std::snprintf(what, sizeof(what), "cublasLtMatmulAlgoConfigGetAttribute(%s)", name);
        p35_set_error_status(what, status);
        return 1;
    }
    if (written != sizeof(value)) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "cublasLtMatmulAlgoConfigGetAttribute(%s) wrote %zu bytes, expected %zu",
                      name, written, sizeof(value));
        return 1;
    }
    *out = static_cast<int64_t>(value);
    return 0;
}

extern "C" {

int p35_bridge_abi_version(void) { return P35_ABI_VERSION; }

size_t p35_plan_info_size(void) { return sizeof(P35PlanInfo); }

const char* p35_last_error(void) { return g_last_error; }

size_t p35_cublaslt_version(void) {
    // The runtime library's own version, not a repository constant and not a
    // build-time macro: this is what actually executed.
    return cublasLtGetVersion();
}

// The frozen shape allowlist, exposed so the Python wrapper can prove that the
// C side and the Python side froze exactly the same five geometries in exactly
// the same order.
size_t p35_shape_count(void) { return P35_SHAPE_COUNT; }

int p35_shape_at(size_t index, int64_t* m, int64_t* n, int64_t* k) {
    p35_clear_error();
    if (m == nullptr || n == nullptr || k == nullptr) {
        p35_set_error("p35_shape_at: every output pointer must be valid");
        return 1;
    }
    if (index >= P35_SHAPE_COUNT) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "p35_shape_at: index %zu is outside the frozen allowlist of %zu shapes",
                      index, P35_SHAPE_COUNT);
        return 1;
    }
    *m = P35_SHAPES[index][0];
    *n = P35_SHAPES[index][1];
    *k = P35_SHAPES[index][2];
    return 0;
}

// Creates the whole cuBLASLt plan for one frozen shape and reports the selected
// algorithm's metadata. This is the only function the wrapper times as
// `setup_time_ms`; it performs no matmul.
int p35_plan_create(int64_t m,
                    int64_t n,
                    int64_t k,
                    const void* a,
                    const void* b,
                    const void* c,
                    void* d,
                    void* stream,
                    P35Plan** out_plan,
                    P35PlanInfo* out_info) {
    p35_clear_error();

    if (out_plan == nullptr || out_info == nullptr) {
        p35_set_error("p35_plan_create: out_plan and out_info must not be null");
        return 1;
    }
    *out_plan = nullptr;
    std::memset(out_info, 0, sizeof(*out_info));

    if (a == nullptr || b == nullptr || c == nullptr || d == nullptr) {
        p35_set_error("p35_plan_create: every operand pointer must be a valid device pointer");
        return 1;
    }

    // (1) The geometry must be one of the five frozen shapes. An arbitrary
    // shape never reaches a descriptor, a heuristic, or a launch.
    const int shape_index = p35_shape_index_of(m, n, k);
    if (shape_index < 0) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "p35_plan_create: (M,N,K)=(%lld,%lld,%lld) is not one of the %zu frozen "
                      "P3.5 shapes; this bridge never plans an arbitrary geometry",
                      static_cast<long long>(m), static_cast<long long>(n),
                      static_cast<long long>(k), P35_SHAPE_COUNT);
        return 1;
    }

    // (2) Independently of the allowlist, prove every derived size is
    // representable before any of them is used.
    if (p35_validate_extents(m, n, k) != 0) {
        return 1;
    }

    // (3) The leading dimensions are derived from the validated shape, never
    // supplied by the caller: A is M x K row-major, B is N x K row-major, and
    // C and D are M x N row-major.
    const int64_t lda = k;
    const int64_t ldb = k;
    const int64_t ldc = n;
    const int64_t ldd = n;

    P35Plan* plan = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    try {
        plan = new (std::nothrow) P35Plan();
        if (plan == nullptr) {
            p35_set_error("p35_plan_create: out of host memory");
            return 1;
        }
        std::memset(plan, 0, sizeof(*plan));
        plan->a = a;
        plan->b = b;
        plan->c = c;
        plan->d = d;
        plan->stream = static_cast<cudaStream_t>(stream);

        cublasStatus_t status = cublasLtCreate(&plan->handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtCreate", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        status = cublasLtMatmulDescCreate(&plan->operation, P35_COMPUTE_TYPE, P35_SCALE_TYPE);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulDescCreate", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        const int32_t transa = static_cast<int32_t>(P35_TRANSA);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transa, sizeof(transa));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulDescSetAttribute(TRANSA)", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        const int32_t transb = static_cast<int32_t>(P35_TRANSB);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transb, sizeof(transb));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulDescSetAttribute(TRANSB)", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        const int32_t pointer_mode = static_cast<int32_t>(P35_POINTER_MODE);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulDescSetAttribute(POINTER_MODE)", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        // The epilogue attribute is documented as uint32_t, unlike the int32_t
        // TRANSA/TRANSB/POINTER_MODE attributes above; the widths are written
        // out separately here rather than assumed to be uniform.
        const uint32_t epilogue = static_cast<uint32_t>(P35_EPILOGUE);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulDescSetAttribute(EPILOGUE)", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        if (p35_create_layout(&plan->layout_a, P35_AB_TYPE, static_cast<uint64_t>(m),
                              static_cast<uint64_t>(k), lda, "A") != 0 ||
            p35_create_layout(&plan->layout_b, P35_AB_TYPE, static_cast<uint64_t>(n),
                              static_cast<uint64_t>(k), ldb, "B") != 0 ||
            p35_create_layout(&plan->layout_c, P35_CD_TYPE, static_cast<uint64_t>(m),
                              static_cast<uint64_t>(n), ldc, "C") != 0 ||
            p35_create_layout(&plan->layout_d, P35_CD_TYPE, static_cast<uint64_t>(m),
                              static_cast<uint64_t>(n), ldd, "D") != 0) {
            (void)p35_plan_release(plan, true);
            return 1;
        }

        const uint32_t alignment_a = p35_pointer_alignment(a);
        const uint32_t alignment_b = p35_pointer_alignment(b);
        const uint32_t alignment_c = p35_pointer_alignment(c);
        const uint32_t alignment_d = p35_pointer_alignment(d);

        status = cublasLtMatmulPreferenceCreate(&preference);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulPreferenceCreate", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        struct PreferenceSetting {
            cublasLtMatmulPreferenceAttributes_t attribute;
            const void* value;
            size_t size;
            const char* name;
        };

        const uint32_t search_mode = static_cast<uint32_t>(P35_SEARCH_MODE);
        const uint64_t workspace_limit = P35_WORKSPACE_LIMIT_BYTES;
        const PreferenceSetting settings[] = {
            {CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode), "SEARCH_MODE"},
            {CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit),
             "MAX_WORKSPACE_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a),
             "MIN_ALIGNMENT_A_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b),
             "MIN_ALIGNMENT_B_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignment_c, sizeof(alignment_c),
             "MIN_ALIGNMENT_C_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignment_d, sizeof(alignment_d),
             "MIN_ALIGNMENT_D_BYTES"},
        };

        for (const PreferenceSetting& setting : settings) {
            status = cublasLtMatmulPreferenceSetAttribute(preference, setting.attribute,
                                                          setting.value, setting.size);
            if (status != CUBLAS_STATUS_SUCCESS) {
                char what[160];
                std::snprintf(what, sizeof(what), "cublasLtMatmulPreferenceSetAttribute(%s)",
                              setting.name);
                p35_set_error_status(what, status);
                (void)p35_preference_release(&preference, true);
                (void)p35_plan_release(plan, true);
                return 1;
            }
        }

        cublasLtMatmulHeuristicResult_t results[P35_HEURISTIC_REQUESTED];
        std::memset(results, 0, sizeof(results));
        int returned = 0;
        status = cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation, plan->layout_a,
                                                plan->layout_b, plan->layout_c, plan->layout_d,
                                                preference, P35_HEURISTIC_REQUESTED, results,
                                                &returned);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulAlgoGetHeuristic", status);
            (void)p35_preference_release(&preference, true);
            (void)p35_plan_release(plan, true);
            return 1;
        }
        if (p35_preference_release(&preference, false) != 0) {
            (void)p35_plan_release(plan, true);
            return 1;
        }
        if (returned <= 0 || returned > P35_HEURISTIC_REQUESTED) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "cublasLtMatmulAlgoGetHeuristic returned %d results for frozen P3.5 "
                          "shape %d (M,N,K)=(%lld,%lld,%lld); no algorithm is available and "
                          "P3.5 never retries with another layout, type, compute mode, "
                          "workspace limit, or API",
                          returned, shape_index, static_cast<long long>(m),
                          static_cast<long long>(n), static_cast<long long>(k));
            (void)p35_plan_release(plan, true);
            return 1;
        }

        // The first supported entry wins. No candidate is executed, timed, or
        // compared here: this is a vendor-heuristic selection, not a search.
        // A different algorithm may naturally win for each shape; the policy
        // that selects it is identical for all five.
        int selected = -1;
        for (int index = 0; index < returned; ++index) {
            if (results[index].state == CUBLAS_STATUS_SUCCESS) {
                selected = index;
                break;
            }
        }
        if (selected < 0) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "none of the %d heuristic results reported CUBLAS_STATUS_SUCCESS for "
                          "frozen P3.5 shape %d",
                          returned, shape_index);
            (void)p35_plan_release(plan, true);
            return 1;
        }

        if (static_cast<uint64_t>(results[selected].workspaceSize) > P35_WORKSPACE_LIMIT_BYTES) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "heuristic result %d needs %zu workspace bytes, above the fixed limit "
                          "of %llu",
                          selected, results[selected].workspaceSize,
                          static_cast<unsigned long long>(P35_WORKSPACE_LIMIT_BYTES));
            (void)p35_plan_release(plan, true);
            return 1;
        }

        plan->algo = results[selected].algo;

        // Re-validate exactly the algorithm that will run, against exactly the
        // descriptors it will run with.
        cublasLtMatmulHeuristicResult_t checked;
        std::memset(&checked, 0, sizeof(checked));
        status = cublasLtMatmulAlgoCheck(plan->handle, plan->operation, plan->layout_a,
                                         plan->layout_b, plan->layout_c, plan->layout_d,
                                         &plan->algo, &checked);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmulAlgoCheck", status);
            (void)p35_plan_release(plan, true);
            return 1;
        }
        if (checked.state != CUBLAS_STATUS_SUCCESS) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "cublasLtMatmulAlgoCheck rejected heuristic result %d with "
                          "cublasStatus_t=%d",
                          selected, static_cast<int>(checked.state));
            (void)p35_plan_release(plan, true);
            return 1;
        }
        if (static_cast<uint64_t>(checked.workspaceSize) > P35_WORKSPACE_LIMIT_BYTES) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "the validated algorithm needs %zu workspace bytes, above the fixed "
                          "limit of %llu",
                          checked.workspaceSize,
                          static_cast<unsigned long long>(P35_WORKSPACE_LIMIT_BYTES));
            (void)p35_plan_release(plan, true);
            return 1;
        }

        // Exactly what the validated algorithm requires, and a null pointer
        // when it requires nothing at all.
        plan->workspace_bytes = checked.workspaceSize;
        if (plan->workspace_bytes > 0) {
            const cudaError_t allocated = cudaMalloc(&plan->workspace, plan->workspace_bytes);
            if (allocated != cudaSuccess) {
                plan->workspace = nullptr;
                p35_set_error_cuda("cudaMalloc(cuBLASLt workspace)", allocated);
                (void)p35_plan_release(plan, true);
                return 1;
            }
        } else {
            plan->workspace = nullptr;
        }

        int64_t algo_id = 0;
        int64_t tile_id = 0;
        int64_t stages_id = 0;
        int64_t split_k = 0;
        int64_t reduction_scheme = 0;
        int64_t cta_swizzling = 0;
        int64_t custom_option = 0;
        int64_t inner_shape_id = 0;
        int64_t cluster_shape_id = 0;

        if (p35_read_algo_config<int32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_ID, "ID", &algo_id) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_TILE_ID, "TILE_ID",
                                           &tile_id) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                           "STAGES_ID", &stages_id) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                           "SPLITK_NUM", &split_k) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                           "REDUCTION_SCHEME", &reduction_scheme) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                           "CTA_SWIZZLING", &cta_swizzling) != 0 ||
            p35_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                           "CUSTOM_OPTION", &custom_option) != 0 ||
            p35_read_algo_config<uint16_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
                                           "INNER_SHAPE_ID", &inner_shape_id) != 0 ||
            p35_read_algo_config<uint16_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                           "CLUSTER_SHAPE_ID", &cluster_shape_id) != 0) {
            (void)p35_plan_release(plan, true);
            return 1;
        }

        out_info->abi_version = P35_ABI_VERSION;
        out_info->cublaslt_version = static_cast<int64_t>(cublasLtGetVersion());

        out_info->shape_index = shape_index;
        out_info->m = m;
        out_info->n = n;
        out_info->k = k;
        out_info->batch_count = P35_BATCH_COUNT;
        out_info->lda = lda;
        out_info->ldb = ldb;
        out_info->ldc = ldc;
        out_info->ldd = ldd;

        out_info->transa = static_cast<int64_t>(P35_TRANSA);
        out_info->transb = static_cast<int64_t>(P35_TRANSB);
        out_info->order_a = static_cast<int64_t>(P35_ORDER);
        out_info->order_b = static_cast<int64_t>(P35_ORDER);
        out_info->order_c = static_cast<int64_t>(P35_ORDER);
        out_info->order_d = static_cast<int64_t>(P35_ORDER);
        out_info->type_a = static_cast<int64_t>(P35_AB_TYPE);
        out_info->type_b = static_cast<int64_t>(P35_AB_TYPE);
        out_info->type_c = static_cast<int64_t>(P35_CD_TYPE);
        out_info->type_d = static_cast<int64_t>(P35_CD_TYPE);
        out_info->compute_type = static_cast<int64_t>(P35_COMPUTE_TYPE);
        out_info->scale_type = static_cast<int64_t>(P35_SCALE_TYPE);
        out_info->pointer_mode = static_cast<int64_t>(P35_POINTER_MODE);
        out_info->epilogue = static_cast<int64_t>(P35_EPILOGUE);

        out_info->search_mode = static_cast<int64_t>(P35_SEARCH_MODE);
        out_info->workspace_limit_bytes = static_cast<int64_t>(P35_WORKSPACE_LIMIT_BYTES);
        out_info->workspace_bytes = static_cast<int64_t>(plan->workspace_bytes);
        out_info->workspace_is_null = (plan->workspace == nullptr) ? 1 : 0;

        out_info->heuristic_requested = P35_HEURISTIC_REQUESTED;
        out_info->heuristic_returned = returned;
        out_info->heuristic_index = selected;

        out_info->alignment_a_bytes = static_cast<int64_t>(alignment_a);
        out_info->alignment_b_bytes = static_cast<int64_t>(alignment_b);
        out_info->alignment_c_bytes = static_cast<int64_t>(alignment_c);
        out_info->alignment_d_bytes = static_cast<int64_t>(alignment_d);

        out_info->algo_id = algo_id;
        out_info->tile_id = tile_id;
        out_info->stages_id = stages_id;
        out_info->split_k = split_k;
        out_info->reduction_scheme = reduction_scheme;
        out_info->cta_swizzling = cta_swizzling;
        out_info->custom_option = custom_option;
        out_info->inner_shape_id = inner_shape_id;
        out_info->cluster_shape_id = cluster_shape_id;

        out_info->waves_count = static_cast<double>(checked.wavesCount);
        out_info->alpha = static_cast<double>(P35_ALPHA);
        out_info->beta = static_cast<double>(P35_BETA);

        *out_plan = plan;
        return 0;
    } catch (...) {
        p35_set_error("p35_plan_create: an unexpected C++ exception was caught at the C boundary");
        (void)p35_preference_release(&preference, true);
        (void)p35_plan_release(plan, true);
        return 1;
    }
}

// Launches exactly one cublasLtMatmul with the selected algorithm on the
// plan's stream. This is the measured path; there is no other one, and no
// fallback to cublasGemmEx, cublasGemmStridedBatchedEx, an ordinary cuBLAS
// GEMM, or a framework operator exists anywhere in this translation unit.
int p35_plan_execute(P35Plan* plan) {
    p35_clear_error();
    if (plan == nullptr) {
        p35_set_error("p35_plan_execute: plan must not be null");
        return 1;
    }
    try {
        const float alpha = P35_ALPHA;
        const float beta = P35_BETA;
        const cublasStatus_t status = cublasLtMatmul(plan->handle,
                                                     plan->operation,
                                                     &alpha,
                                                     plan->a,
                                                     plan->layout_a,
                                                     plan->b,
                                                     plan->layout_b,
                                                     &beta,
                                                     plan->c,
                                                     plan->layout_c,
                                                     plan->d,
                                                     plan->layout_d,
                                                     &plan->algo,
                                                     plan->workspace,
                                                     plan->workspace_bytes,
                                                     plan->stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p35_set_error_status("cublasLtMatmul", status);
            return 1;
        }
        return 0;
    } catch (...) {
        p35_set_error("p35_plan_execute: an unexpected C++ exception was caught at the C boundary");
        return 1;
    }
}

// Synchronizes the plan's stream. Kept here so the wrapper's host-clock
// timings bound completed GPU work rather than an asynchronous launch.
int p35_stream_synchronize(P35Plan* plan) {
    p35_clear_error();
    if (plan == nullptr) {
        p35_set_error("p35_stream_synchronize: plan must not be null");
        return 1;
    }
    try {
        const cudaError_t status = cudaStreamSynchronize(plan->stream);
        if (status != cudaSuccess) {
            p35_set_error_cuda("cudaStreamSynchronize", status);
            return 1;
        }
        return 0;
    } catch (...) {
        p35_set_error("p35_stream_synchronize: an unexpected C++ exception was caught");
        return 1;
    }
}

// Releases every cuBLASLt object and the workspace of one shape's plan, so no
// shape-owned resource outlives the shape that created it.
int p35_plan_destroy(P35Plan* plan) {
    p35_clear_error();
    try {
        return p35_plan_release(plan, false);
    } catch (...) {
        p35_set_error("p35_plan_destroy: an unexpected C++ exception was caught at the C boundary");
        return 1;
    }
}

}  // extern "C"
