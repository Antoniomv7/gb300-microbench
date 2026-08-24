#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include <cuda_runtime.h>

inline cudaDeviceProp benchmark_device_properties() {
    int count = 0;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess && count == 1) status = cudaSetDevice(0);
    if (status == cudaSuccess && count == 1) status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess || count != 1 || properties.major != 10 || properties.minor != 3) {
        std::fprintf(stderr, "benchmark: one Blackwell sm_103 GPU is required (%s)\n",
                     cudaGetErrorString(status));
        std::exit(1);
    }
    return properties;
}

inline std::string now_utc_iso8601() {
    const std::time_t timestamp = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&timestamp, &utc);
    char output[32];
    std::strftime(output, sizeof(output), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return output;
}

inline bool parse_int_arg(const std::string& input, int64_t* output) {
    if (input.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(input.c_str(), &end, 10);
    if (errno != 0 || end == input.c_str() || *end != '\0') return false;
    *output = static_cast<int64_t>(value);
    return true;
}
