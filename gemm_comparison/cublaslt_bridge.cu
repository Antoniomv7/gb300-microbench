// cuBLASLt baseline: row-major BF16 A·Bᵀ, FP32 accumulation and output.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>

namespace {

constexpr uint64_t kWorkspaceLimit = 64ULL * 1024 * 1024;
constexpr int kRequestedAlgorithms = 32;
thread_local char g_error[256] = {};

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
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm = {};
    const void* a = nullptr;
    const void* b = nullptr;
    const void* c = nullptr;
    void* d = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = nullptr;

    ~Plan() {
        if (workspace) cudaFree(workspace);
        if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
        if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
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

int create_layout(cublasLtMatrixLayout_t* layout, cudaDataType_t type,
                  int64_t rows, int64_t columns, int64_t leading_dimension) {
    CHECK_LT(cublasLtMatrixLayoutCreate(layout, type, rows, columns, leading_dimension));
    const int32_t order = static_cast<int32_t>(CUBLASLT_ORDER_ROW);
    const int32_t batch_count = 1;
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(
        *layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_LT(cublasLtMatrixLayoutSetAttribute(
        *layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    return 0;
}

uint32_t alignment(const void* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    uint32_t bytes = 256;
    while (bytes > 1 && address % bytes != 0) bytes /= 2;
    return bytes;
}

}  // namespace

extern "C" {

struct GbPlanInfo {
    int64_t workspace_bytes;
    int64_t heuristic_index;
    int64_t algorithm_id;
};

const char* gb_last_error() { return g_error; }

int gb_plan_create(int64_t m, int64_t n, int64_t k, const void* a, const void* b,
                   const void* c, void* d, void* stream, Plan** result, GbPlanInfo* info) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c || !d || !result || !info)
        return fail("invalid cuBLASLt plan arguments");

    std::unique_ptr<Plan> plan(new (std::nothrow) Plan());
    if (!plan) return fail("cannot allocate the cuBLASLt plan");
    plan->a = a;
    plan->b = b;
    plan->c = c;
    plan->d = d;
    plan->stream = static_cast<cudaStream_t>(stream);

    CHECK_LT(cublasLtCreate(&plan->handle));
    CHECK_LT(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const int32_t transa = CUBLAS_OP_N;
    const int32_t transb = CUBLAS_OP_T;
    const int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    const uint32_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (create_layout(&plan->a_layout, CUDA_R_16BF, m, k, k) ||
        create_layout(&plan->b_layout, CUDA_R_16BF, n, k, k) ||
        create_layout(&plan->c_layout, CUDA_R_32F, m, n, n) ||
        create_layout(&plan->d_layout, CUDA_R_32F, m, n, n))
        return 1;

    Preference preference;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&preference.value));
    const uint32_t search_mode = CUBLASLT_SEARCH_BEST_FIT;
    const uint32_t alignments[] = {alignment(a), alignment(b), alignment(c), alignment(d)};
    const struct {
        cublasLtMatmulPreferenceAttributes_t attribute;
        const void* value;
        size_t bytes;
    } settings[] = {
        {CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode)},
        {CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceLimit, sizeof(kWorkspaceLimit)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignments[0], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignments[1], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignments[2], sizeof(uint32_t)},
        {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignments[3], sizeof(uint32_t)},
    };
    for (const auto& setting : settings)
        CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
            preference.value, setting.attribute, setting.value, setting.bytes));

    cublasLtMatmulHeuristicResult_t candidates[kRequestedAlgorithms] = {};
    int returned = 0;
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(
        plan->handle, plan->operation, plan->a_layout, plan->b_layout,
        plan->c_layout, plan->d_layout, preference.value, kRequestedAlgorithms,
        candidates, &returned));

    int selected = -1;
    for (int index = 0; index < returned; ++index) {
        if (candidates[index].state == CUBLAS_STATUS_SUCCESS) {
            selected = index;
            break;
        }
    }
    if (selected < 0) return fail("cuBLASLt did not return a supported algorithm");
    plan->algorithm = candidates[selected].algo;
    cublasLtMatmulHeuristicResult_t checked = {};
    CHECK_LT(cublasLtMatmulAlgoCheck(
        plan->handle, plan->operation, plan->a_layout, plan->b_layout,
        plan->c_layout, plan->d_layout, &plan->algorithm, &checked));
    if (checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > kWorkspaceLimit)
        return fail("the selected cuBLASLt algorithm exceeds the workspace limit");

    plan->workspace_bytes = checked.workspaceSize;
    if (plan->workspace_bytes && cudaMalloc(&plan->workspace, plan->workspace_bytes) != cudaSuccess)
        return fail("cannot allocate the cuBLASLt workspace");

    int32_t algorithm_id = -1;
    size_t written = 0;
    CHECK_LT(cublasLtMatmulAlgoConfigGetAttribute(
        &plan->algorithm, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
        sizeof(algorithm_id), &written));
    info->workspace_bytes = static_cast<int64_t>(plan->workspace_bytes);
    info->heuristic_index = selected;
    info->algorithm_id = algorithm_id;
    *result = plan.release();
    return 0;
}

int gb_plan_execute(Plan* plan) {
    if (!plan) return fail("the cuBLASLt plan is not initialized");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_LT(cublasLtMatmul(plan->handle, plan->operation, &alpha,
                            plan->a, plan->a_layout, plan->b, plan->b_layout,
                            &beta, plan->c, plan->c_layout, plan->d, plan->d_layout,
                            &plan->algorithm, plan->workspace, plan->workspace_bytes,
                            plan->stream));
    return 0;
}

int gb_plan_destroy(Plan* plan) {
    delete plan;
    return 0;
}

}  // extern "C"
