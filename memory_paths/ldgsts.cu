// LDGSTS global-to-shared bandwidth.

#define MEMORY_METHOD "ldgsts"
#include "memory_common.cuh"

namespace {

template <int COPIES>
__device__ __forceinline__ void emit_stage_cp_async(
        const uint4* __restrict__ tile_src, unsigned char* __restrict__ slot, int tid) {
#pragma unroll
    for (int c = 0; c < COPIES; ++c) {
        const int vec = tid + c * kThreadsPerCta;
        const uint4* src_ptr = tile_src + vec;
        unsigned char* dst_ptr = slot + static_cast<size_t>(vec) * kVectorBytes;
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr), "l"(src_ptr), "n"(kVectorBytes)
                     : "memory");
    }
}

template <int STAGES, int COPIES>
__global__ void ldgsts_validate_kernel(
        const uint4* __restrict__ g_src,
        int64_t tiles_per_cta,
        unsigned long long* __restrict__ g_mismatch_count) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kElemsPerTile = static_cast<int64_t>(kThreadsPerCta) * COPIES;
    constexpr int64_t kStageBytes = kElemsPerTile * kVectorBytes;

    const int tid = threadIdx.x;
    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    unsigned long long local_mismatches = 0;

    auto consume = [&](int64_t consume_s) {
        const int64_t tile_idx = cta_tile_base + consume_s;
        const int64_t tile_vec_base = tile_idx * kElemsPerTile;
        unsigned char* slot = smem + (consume_s % STAGES) * kStageBytes;
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
            const int vec = tid + c * kThreadsPerCta;
            const uint4 got = *reinterpret_cast<uint4*>(slot + static_cast<size_t>(vec) * kVectorBytes);
            const uint4 want = expected_vector(tile_vec_base + vec);
            if (got.x != want.x || got.y != want.y || got.z != want.z || got.w != want.w) {
                ++local_mismatches;
            }
        }
    };

    for (int64_t s = 0; s < tiles_per_cta; ++s) {
        const uint4* tile_src = g_src + (cta_tile_base + s) * kElemsPerTile;
        unsigned char* slot = smem + (s % STAGES) * kStageBytes;
        emit_stage_cp_async<COPIES>(tile_src, slot, tid);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        if (s >= STAGES - 1) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1) : "memory");
            __syncthreads();
            consume(s - (STAGES - 1));
            __syncthreads();
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    int64_t drain_start = tiles_per_cta - STAGES + 1;
    if (drain_start < 0) drain_start = 0;
    for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
        consume(consume_s);
    }
    __syncthreads();

    if (local_mismatches != 0) {
        atomicAdd(g_mismatch_count, local_mismatches);
    }
}

template <int STAGES, int COPIES>
__global__ void ldgsts_benchmark_kernel(
        const uint4* __restrict__ g_src,
        uint32_t* __restrict__ g_sink,
        int64_t tiles_per_cta,
        int64_t passes,
        int64_t rotation_base) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kElemsPerTile = static_cast<int64_t>(kThreadsPerCta) * COPIES;
    constexpr int64_t kStageBytes = kElemsPerTile * kVectorBytes;

    const int tid = threadIdx.x;
    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    uint32_t sink_acc = 0;

    for (int64_t p = 0; p < passes; ++p) {
        const int64_t rotation = (rotation_base + p) % tiles_per_cta;

        auto touch = [&](int64_t consume_s) {
            unsigned char* slot = smem + (consume_s % STAGES) * kStageBytes;
            sink_acc ^= *reinterpret_cast<uint32_t*>(slot + static_cast<size_t>(tid) * kVectorBytes);
        };

        // Keep asynchronous copy groups in flight while consuming older tiles.
        for (int64_t s = 0; s < tiles_per_cta; ++s) {
            const int64_t tile_idx = cta_tile_base + ((rotation + s) % tiles_per_cta);
            const uint4* tile_src = g_src + tile_idx * kElemsPerTile;
            unsigned char* slot = smem + (s % STAGES) * kStageBytes;
            emit_stage_cp_async<COPIES>(tile_src, slot, tid);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            if (s >= STAGES - 1) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1) : "memory");
                __syncthreads();
                touch(s - (STAGES - 1));
                __syncthreads();
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        int64_t drain_start = tiles_per_cta - STAGES + 1;
        if (drain_start < 0) drain_start = 0;
        for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
            touch(consume_s);
        }
        __syncthreads();
    }

    g_sink[static_cast<int64_t>(blockIdx.x) * kThreadsPerCta + tid] = sink_acc;
}

template <int STAGES, int COPIES>
RunStatus run_specialization(const GpuInfo& gpu, const Specialization& spec,
                             const WorkingSetPlan& working_set, const CliConfig& config) {
    return run_memory_case(
        gpu, spec, working_set, config, ldgsts_validate_kernel<STAGES, COPIES>,
        ldgsts_benchmark_kernel<STAGES, COPIES>, 0,
        [](uint4*) { return RunStatus::kOk; },
        [](uint4* source, unsigned long long* mismatches, int64_t tiles, int blocks, size_t smem) {
            ldgsts_validate_kernel<STAGES, COPIES><<<blocks, kThreadsPerCta, smem>>>(
                source, tiles, mismatches);
        },
        [](uint4* source, uint32_t* sink, int64_t tiles, int64_t passes, int64_t rotation,
           int blocks, size_t smem) {
            ldgsts_benchmark_kernel<STAGES, COPIES><<<blocks, kThreadsPerCta, smem>>>(
                source, sink, tiles, passes, rotation);
        });
}

RunStatus dispatch_run(const GpuInfo& gpu, const Specialization& spec,
                       const WorkingSetPlan& working_set, const CliConfig& config) {
#define MEMORY_CASE(STAGES, BIF, COPIES)                                      \
    if (config.stages == STAGES && config.bif_kib == BIF)                     \
        return run_specialization<STAGES, COPIES>(gpu, spec, working_set, config)
    MEMORY_CASE(2, 16, 4); MEMORY_CASE(2, 32, 8); MEMORY_CASE(2, 64, 16);
    MEMORY_CASE(4, 16, 2); MEMORY_CASE(4, 32, 4); MEMORY_CASE(4, 64, 8);
    MEMORY_CASE(8, 16, 1); MEMORY_CASE(8, 32, 2); MEMORY_CASE(8, 64, 4);
#undef MEMORY_CASE
    fail("unsupported configuration");
}

}  // namespace

int main(int argc, char** argv) {
    CliConfig config;
    std::string error;
    if (!parse_cli(argc, argv, &config, &error)) {
        std::fprintf(stderr, "ldgsts: %s\n", error.c_str());
        print_usage(stderr);
        return 2;
    }
    if (config.help) { print_usage(stdout); return 0; }
    const GpuInfo gpu = query_gpu_info();
    const WorkingSetPlan working_set = plan_working_set(gpu, config.working_set_mib);
    if (config.run_kind == "benchmark" && working_set.working_set_bytes <= 2 * gpu.l2_bytes)
        fail("working set must exceed twice the L2 cache size");
    return dispatch_run(gpu, find_spec(config.stages, config.bif_kib), working_set, config)
        == RunStatus::kOk ? 0 : 1;
}
