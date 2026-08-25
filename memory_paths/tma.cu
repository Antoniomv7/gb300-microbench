// TMA global-to-shared bandwidth.

#include <cudaTypedefs.h>
#include <cuda/ptx>

#define MEMORY_METHOD "tma"
#include "memory_common.cuh"

namespace {

__device__ __forceinline__ bool tma_elect_leader() {
    bool is_leader = false;
    if (threadIdx.x < 32) {
        is_leader = cuda::ptx::elect_sync(0xFFFFFFFFu);
    }
    return is_leader;
}

template <int STAGES>
__device__ __forceinline__ void tma_init_barriers(uint64_t* bars) {
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            cuda::ptx::mbarrier_init(&bars[s], 1u);
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();
}

__device__ __forceinline__ void tma_issue_stage(
        bool is_leader, unsigned char* dst_slot, const CUtensorMap* tensor_map,
        int32_t coord_y, uint64_t* bar, uint32_t stage_bytes) {
    if (is_leader) {
        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta,
                                              cuda::ptx::space_shared, bar, stage_bytes);
        const int32_t coords[2] = {0, coord_y};
        cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared, cuda::ptx::space_global,
                                         dst_slot, tensor_map, coords, bar);
    }
}

__device__ __forceinline__ void tma_wait_stage(uint64_t* bar, uint32_t& parity) {
    while (!cuda::ptx::mbarrier_try_wait_parity(bar, parity)) {
    }
    parity ^= 1u;
}

__device__ __forceinline__ void tma_invalidate_barrier(uint64_t* bar) {
    asm volatile("mbarrier.inval.shared.b64 [%0];"
                 :
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar)))
                 : "memory");
}

template <int STAGES>
__device__ __forceinline__ void tma_invalidate_barriers(bool is_leader, uint64_t* bars) {
    if (is_leader) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            tma_invalidate_barrier(&bars[s]);
        }
    }
    __syncthreads();
}

template <int STAGES, int COPIES>
__global__ void tma_validate_kernel(
        const __grid_constant__ CUtensorMap tensor_map,
        int64_t tiles_per_cta,
        unsigned long long* __restrict__ g_mismatch_count) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kStageBytes = compute_stage_bytes(COPIES);
    constexpr int64_t kElemsPerTile = kStageBytes / kVectorBytes;
    constexpr int64_t kPayloadBytes = static_cast<int64_t>(STAGES) * kStageBytes;
    constexpr int64_t kTileHeight = compute_tile_height(kStageBytes);

    unsigned char* payload = smem;
    // Each pipeline slot has an mbarrier for its asynchronous TMA transfer.
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + kPayloadBytes);

    const int tid = threadIdx.x;
    const bool is_leader = tma_elect_leader();
    tma_init_barriers<STAGES>(bars);

    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    unsigned long long local_mismatches = 0;
    uint32_t parity[STAGES];
#pragma unroll
    for (int i = 0; i < STAGES; ++i) parity[i] = 0u;

    auto issue = [&](int64_t s) {
        const int slot = static_cast<int>(s % STAGES);
        unsigned char* dst = payload + static_cast<size_t>(slot) * kStageBytes;
        const int64_t tile_idx = cta_tile_base + s;
        tma_issue_stage(is_leader, dst, &tensor_map,
                         static_cast<int32_t>(tile_idx * kTileHeight), &bars[slot],
                         static_cast<uint32_t>(kStageBytes));
    };

    auto consume = [&](int64_t consume_s) {
        const int slot = static_cast<int>(consume_s % STAGES);
        const int64_t tile_idx = cta_tile_base + consume_s;
        const int64_t tile_vec_base = tile_idx * kElemsPerTile;
        unsigned char* slot_ptr = payload + static_cast<size_t>(slot) * kStageBytes;
#pragma unroll
        for (int c = 0; c < COPIES; ++c) {
            const int vec = tid + c * kThreadsPerCta;
            const uint4 got = *reinterpret_cast<uint4*>(slot_ptr + static_cast<size_t>(vec) * kVectorBytes);
            const uint4 want = expected_vector(tile_vec_base + vec);
            if (got.x != want.x || got.y != want.y || got.z != want.z || got.w != want.w) {
                ++local_mismatches;
            }
        }
    };

    for (int64_t s = 0; s < tiles_per_cta; ++s) {
        issue(s);
        if (s >= STAGES - 1) {
            const int64_t consume_s = s - (STAGES - 1);
            tma_wait_stage(&bars[consume_s % STAGES], parity[consume_s % STAGES]);
            consume(consume_s);
            __syncthreads();
        }
    }
    int64_t drain_start = tiles_per_cta - STAGES + 1;
    if (drain_start < 0) drain_start = 0;
    for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
        tma_wait_stage(&bars[consume_s % STAGES], parity[consume_s % STAGES]);
        consume(consume_s);
    }
    __syncthreads();

    tma_invalidate_barriers<STAGES>(is_leader, bars);

    if (local_mismatches != 0) {
        atomicAdd(g_mismatch_count, local_mismatches);
    }
}

template <int STAGES, int COPIES>
__global__ void tma_benchmark_kernel(
        const __grid_constant__ CUtensorMap tensor_map,
        uint32_t* __restrict__ g_sink,
        int64_t tiles_per_cta,
        int64_t passes,
        int64_t rotation_base) {
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int64_t kStageBytes = compute_stage_bytes(COPIES);
    constexpr int64_t kPayloadBytes = static_cast<int64_t>(STAGES) * kStageBytes;
    constexpr int64_t kTileHeight = compute_tile_height(kStageBytes);

    unsigned char* payload = smem;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + kPayloadBytes);

    const int tid = threadIdx.x;
    const bool is_leader = tma_elect_leader();
    tma_init_barriers<STAGES>(bars);

    const int64_t cta_tile_base = static_cast<int64_t>(blockIdx.x) * tiles_per_cta;
    uint32_t sink_acc = 0;
    uint32_t parity[STAGES];
#pragma unroll
    for (int i = 0; i < STAGES; ++i) parity[i] = 0u;

    for (int64_t p = 0; p < passes; ++p) {
        const int64_t rotation = (rotation_base + p) % tiles_per_cta;

        auto issue = [&](int64_t s) {
            const int slot = static_cast<int>(s % STAGES);
            unsigned char* dst = payload + static_cast<size_t>(slot) * kStageBytes;
            const int64_t tile_idx = cta_tile_base + ((rotation + s) % tiles_per_cta);
            tma_issue_stage(is_leader, dst, &tensor_map,
                             static_cast<int32_t>(tile_idx * kTileHeight), &bars[slot],
                             static_cast<uint32_t>(kStageBytes));
        };

        auto touch = [&](int64_t consume_s) {
            const int slot = static_cast<int>(consume_s % STAGES);
            unsigned char* slot_ptr = payload + static_cast<size_t>(slot) * kStageBytes;
            sink_acc ^= *reinterpret_cast<uint32_t*>(slot_ptr + static_cast<size_t>(tid) * kVectorBytes);
        };

        for (int64_t s = 0; s < tiles_per_cta; ++s) {
            issue(s);
            if (s >= STAGES - 1) {
                const int64_t consume_s = s - (STAGES - 1);
                tma_wait_stage(&bars[consume_s % STAGES], parity[consume_s % STAGES]);
                touch(consume_s);
                __syncthreads();
            }
        }
        int64_t drain_start = tiles_per_cta - STAGES + 1;
        if (drain_start < 0) drain_start = 0;
        for (int64_t consume_s = drain_start; consume_s < tiles_per_cta; ++consume_s) {
            tma_wait_stage(&bars[consume_s % STAGES], parity[consume_s % STAGES]);
            touch(consume_s);
        }
        __syncthreads();
    }

    tma_invalidate_barriers<STAGES>(is_leader, bars);

    g_sink[static_cast<int64_t>(blockIdx.x) * kThreadsPerCta + tid] = sink_acc;
}

PFN_cuTensorMapEncodeTiled_v12000 load_tensor_map_encoder() {
    void* address = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &address, 12000,
                                          cudaEnableDefault, &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess || !address)
        fail("cuTensorMapEncodeTiled is unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
}

RunStatus build_tensor_map(PFN_cuTensorMapEncodeTiled_v12000 encode, void* source,
                            int64_t bytes, int tile_height, CUtensorMap* tensor_map) {
    const cuuint64_t dimensions[] = {kTileWidthElements,
                                      static_cast<cuuint64_t>(bytes / kTileWidthBytes)};
    const cuuint64_t strides[] = {kTileWidthBytes};
    const cuuint32_t box[] = {kTileWidthElements, static_cast<cuuint32_t>(tile_height)};
    const cuuint32_t element_strides[] = {1, 1};
    const CUresult status = encode(tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, source,
                                    dimensions, strides, box, element_strides,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return status == CUDA_SUCCESS ? RunStatus::kOk : RunStatus::kCudaError;
}

template <int STAGES, int COPIES>
RunStatus run_specialization(const GpuInfo& gpu, const Specialization& spec,
                             const WorkingSetPlan& working_set, const CliConfig& config,
                             PFN_cuTensorMapEncodeTiled_v12000 encode) {
    CUtensorMap tensor_map{};
    return run_memory_case(
        gpu, spec, working_set, config, tma_validate_kernel<STAGES, COPIES>,
        tma_benchmark_kernel<STAGES, COPIES>, STAGES * sizeof(uint64_t),
        [&](uint4* source) {
            return build_tensor_map(encode, source, working_set.working_set_bytes,
                                    spec.tile_height, &tensor_map);
        },
        [&](uint4*, unsigned long long* mismatches, int64_t tiles, int blocks, size_t smem) {
            tma_validate_kernel<STAGES, COPIES><<<blocks, kThreadsPerCta, smem>>>(
                tensor_map, tiles, mismatches);
        },
        [&](uint4*, uint32_t* sink, int64_t tiles, int64_t passes, int64_t rotation,
            int blocks, size_t smem) {
            tma_benchmark_kernel<STAGES, COPIES><<<blocks, kThreadsPerCta, smem>>>(
                tensor_map, sink, tiles, passes, rotation);
        });
}

RunStatus dispatch_run(const GpuInfo& gpu, const Specialization& spec,
                       const WorkingSetPlan& working_set, const CliConfig& config,
                       PFN_cuTensorMapEncodeTiled_v12000 encode) {
#define MEMORY_CASE(STAGES, BIF, COPIES)                                      \
    if (config.stages == STAGES && config.bif_kib == BIF)                     \
        return run_specialization<STAGES, COPIES>(gpu, spec, working_set, config, encode)
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
        std::fprintf(stderr, "tma: %s\n", error.c_str());
        print_usage(stderr);
        return 2;
    }
    if (config.help) { print_usage(stdout); return 0; }
    const GpuInfo gpu = query_gpu_info();
    const WorkingSetPlan working_set = plan_working_set(gpu, config.working_set_mib);
    if (config.run_kind == "benchmark" && working_set.working_set_bytes <= 2 * gpu.l2_bytes)
        fail("working set must exceed twice the L2 cache size");
    return dispatch_run(gpu, find_spec(config.stages, config.bif_kib), working_set, config,
                        load_tensor_map_encoder()) == RunStatus::kOk ? 0 : 1;
}
