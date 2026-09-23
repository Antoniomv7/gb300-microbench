// cuBLASLt baseline for the precision comparison: D = A·Bᵀ for K-major BF16, FP8 E4M3 or
// block-scaled NVFP4 operands with FP32 accumulation, alpha = 1 and beta = 0.
//
// A (M×K) and B (N×K) are row-major with K contiguous and D (M×N) is row-major. cuBLASLt is
// column-major, so the plan computes Dᵀ = B·Aᵀ as the canonical "TN" problem required by
// FP8 and block-scaled kernels: cuBLASLt's A is our B, cuBLASLt's B is our A.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

// Selected-algorithm record, mirrored by ctypes in precision_comparison/cublaslt_precision.py.
// It lives outside the anonymous namespace so the extern "C" entry points keep external linkage.
struct GbpPlanInfo {
    int32_t returned_algorithms;
    int32_t selected_index;
    int32_t algorithm_id;
    uint32_t tile_id;
    uint32_t stages_id;
    int32_t splitk_num;
    uint32_t reduction_scheme;
    uint32_t cta_swizzling;
    uint32_t custom_option;
    uint16_t inner_shape_id;
    uint16_t cluster_shape_id;
    uint64_t workspace_bytes;
    float waves_count;
    int32_t check_status;
    uint64_t check_workspace_bytes;
    uint64_t algorithm_data[8];
};

namespace {

constexpr int kRequestedAlgorithms = 32;
thread_local char g_error[256] = {};

enum Format : int32_t { kBFloat16 = 0, kFloat8E4M3 = 1, kNVFloat4 = 2 };
enum Output : int32_t { kOutputFloat32 = 0, kOutputBFloat16 = 1 };

int fail(const char* operation, cublasStatus_t status) {
    std::snprintf(g_error, sizeof(g_error), "%s: cuBLAS status %d",
                  operation, static_cast<int>(status));
    return 1;
}

int fail(const char* message) {
    std::snprintf(g_error, sizeof(g_error), "%s", message);
    return 1;
}

#define CHECK_LT(expression)                       \
    do {                                          \
        const cublasStatus_t status = expression; \
        if (status != CUBLAS_STATUS_SUCCESS) {     \
            return fail(#expression, status);     \
        }                                         \
    } while (0)

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm = {};
    const void* a = nullptr;  // cuBLASLt A: our B
    const void* b = nullptr;  // cuBLASLt B: our A
    void* d = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = nullptr;

    ~Plan() {
        if (workspace) cudaFree(workspace);
        if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
        if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
};

struct Preference {
    cublasLtMatmulPreference_t value = nullptr;
    ~Preference() {
        if (value) cublasLtMatmulPreferenceDestroy(value);
    }
};

uint32_t alignment(const void* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    uint32_t bytes = 256;
    while (bytes > 1 && address % bytes != 0) bytes /= 2;
    return bytes;
}

template <typename T>
int config(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t attribute,
           T* value) {
    size_t written = 0;
    CHECK_LT(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, value, sizeof(T), &written));
    return 0;
}

}  // namespace

extern "C" {

const char* gbp_last_error() { return g_error; }

size_t gbp_cublaslt_version() { return cublasLtGetVersion(); }

int gbp_plan_create(int32_t format, int32_t output, int64_t m, int64_t n, int64_t k,
                    const void* a, const void* b, const void* a_scale, const void* b_scale,
                    void* d, void* stream, uint64_t workspace_limit, void** result,
                    GbpPlanInfo* info) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !d || !result || !info)
        return fail("invalid cuBLASLt plan arguments");
    if (format != kBFloat16 && format != kFloat8E4M3 && format != kNVFloat4)
        return fail("unknown input format");
    if (output != kOutputFloat32 && output != kOutputBFloat16) return fail("unknown output type");
    if (format == kNVFloat4 && (!a_scale || !b_scale))
        return fail("NVFP4 requires both block-scale tensors");
    std::memset(info, 0, sizeof(*info));

    std::unique_ptr<Plan> plan(new (std::nothrow) Plan());
    if (!plan) return fail("cannot allocate the cuBLASLt plan");
    plan->a = b;
    plan->b = a;
    plan->d = d;
    plan->stream = static_cast<cudaStream_t>(stream);

    const cudaDataType_t input = format == kBFloat16      ? CUDA_R_16BF
                                 : format == kFloat8E4M3 ? CUDA_R_8F_E4M3
                                                         : CUDA_R_4F_E2M1;
    const cudaDataType_t output_type = output == kOutputFloat32 ? CUDA_R_32F : CUDA_R_16BF;
    CHECK_LT(cublasLtCreate(&plan->handle));
    CHECK_LT(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const int32_t transa = CUBLAS_OP_T;
    const int32_t transb = CUBLAS_OP_N;
    const int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    const uint32_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    const int8_t fast_accumulation = 0;
    const struct {
        cublasLtMatmulDescAttributes_t attribute;
        const void* value;
        size_t bytes;
    } settings[] = {
        {CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)},
        {CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)},
        {CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)},
        {CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)},
        {CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accumulation, sizeof(fast_accumulation)},
    };
    for (const auto& setting : settings)
        CHECK_LT(cublasLtMatmulDescSetAttribute(plan->operation, setting.attribute,
                                                setting.value, setting.bytes));
    if (format == kNVFloat4) {
        // One UE4M3 scale per 16 consecutive K elements of each operand row.
        const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CHECK_LT(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof(b_scale)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    }

    // Column-major K×N and K×M operands with leading dimension K; D is column-major N×M.
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->a_layout, input, k, n, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->b_layout, input, k, m, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->d_layout, output_type, n, m, n));

    Preference preference;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&preference.value));
    const uint32_t search_mode = CUBLASLT_SEARCH_BEST_FIT;
    const uint32_t alignments[] = {alignment(plan->a), alignment(plan->b), alignment(d)};
    const struct {
        cublasLtMatmulPreferenceAttributes_t attribute;
        const void* value;
        size_t bytes;
    } preferences[] = {
        {CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode)},
        {CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignments[0], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignments[1], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignments[2], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignments[2], sizeof(uint32_t)},
    };
    for (const auto& setting : preferences)
        CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
            preference.value, setting.attribute, setting.value, setting.bytes));

    cublasLtMatmulHeuristicResult_t candidates[kRequestedAlgorithms] = {};
    int returned = 0;
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(
        plan->handle, plan->operation, plan->a_layout, plan->b_layout, plan->d_layout,
        plan->d_layout, preference.value, kRequestedAlgorithms, candidates, &returned));
    info->returned_algorithms = returned;
    info->selected_index = -1;
    // Same policy as the campaign baseline: the first supported heuristic, no timed search.
    for (int index = 0; index < returned; ++index) {
        if (candidates[index].state == CUBLAS_STATUS_SUCCESS) {
            info->selected_index = index;
            break;
        }
    }
    if (info->selected_index < 0) return fail("cuBLASLt did not return a supported algorithm");
    const auto& selected = candidates[info->selected_index];
    plan->algorithm = selected.algo;
    plan->workspace_bytes = selected.workspaceSize;
    info->workspace_bytes = selected.workspaceSize;
    info->waves_count = selected.wavesCount;
    std::memcpy(info->algorithm_data, plan->algorithm.data, sizeof(info->algorithm_data));
    if (config(plan->algorithm, CUBLASLT_ALGO_CONFIG_ID, &info->algorithm_id) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, &info->tile_id) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, &info->stages_id) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &info->splitk_num) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &info->reduction_scheme) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &info->cta_swizzling) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &info->custom_option) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &info->inner_shape_id) ||
        config(plan->algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &info->cluster_shape_id))
        return 1;

    // Confirm independently that the selected algorithm accepts this exact output type.
    cublasLtMatmulHeuristicResult_t check = {};
    const cublasStatus_t status = cublasLtMatmulAlgoCheck(
        plan->handle, plan->operation, plan->a_layout, plan->b_layout, plan->d_layout,
        plan->d_layout, &plan->algorithm, &check);
    info->check_status = static_cast<int32_t>(status);
    info->check_workspace_bytes = check.workspaceSize;
    if (status != CUBLAS_STATUS_SUCCESS) return fail("cublasLtMatmulAlgoCheck", status);

    if (plan->workspace_bytes && cudaMalloc(&plan->workspace, plan->workspace_bytes) != cudaSuccess)
        return fail("cannot allocate the cuBLASLt workspace");
    *result = plan.release();
    return 0;
}

int gbp_plan_execute(void* handle) {
    auto* plan = static_cast<Plan*>(handle);
    if (!plan) return fail("the cuBLASLt plan is not initialized");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // beta = 0: C is not read, so it aliases D.
    CHECK_LT(cublasLtMatmul(plan->handle, plan->operation, &alpha,
                            plan->a, plan->a_layout, plan->b, plan->b_layout,
                            &beta, plan->d, plan->d_layout, plan->d, plan->d_layout,
                            &plan->algorithm, plan->workspace, plan->workspace_bytes,
                            plan->stream));
    return 0;
}

int gbp_plan_destroy(void* handle) {
    delete static_cast<Plan*>(handle);
    return 0;
}

}  // extern "C"
