#include "renderer/queue_compaction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace
{
__global__ void compactPattern(uint32_t* output, uint32_t* outputCount, uint32_t capacity, uint32_t pattern) {
    uint32_t index = threadIdx.x;
    bool active = pattern == 0 ? (index % 3 != 0) : index < 47;
    appendWarpCompacted(active, index, output, outputCount, capacity);
}

bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "queue_compaction_tests: " << message << '\n';
    return condition;
}

bool validateCompacted(const std::vector<uint32_t>& output, uint32_t count, uint32_t capacity, uint32_t pattern) {
    bool valid = true;
    valid &= check(count == (pattern == 0 ? 42u : 47u), "compaction count is incorrect");

    std::unordered_set<uint32_t> seen;
    uint32_t storedCount = std::min(count, capacity);
    for (uint32_t i = 0; i < storedCount; ++i) {
        uint32_t value = output[i];
        bool active = pattern == 0 ? (value < 64 && value % 3 != 0) : value < 47;
        valid &= check(active, "compaction emitted an inactive item");
        valid &= check(seen.insert(value).second, "compaction emitted a duplicate item");
    }
    return valid;
}
} // namespace

int main() {
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || deviceStatus == cudaErrorInsufficientDriver || deviceCount == 0) {
        std::cout << "queue_compaction_tests: skipped (no CUDA device)\n";
        return 0;
    }
    if (deviceStatus != cudaSuccess) {
        std::cerr << "queue_compaction_tests: CUDA device query failed: " << cudaGetErrorString(deviceStatus) << '\n';
        return 1;
    }

    bool valid = true;

    {
        constexpr uint32_t capacity = 64;
        uint32_t* deviceOutput = nullptr;
        uint32_t* deviceCount = nullptr;
        cudaMalloc(&deviceOutput, sizeof(uint32_t) * capacity);
        cudaMalloc(&deviceCount, sizeof(uint32_t));
        cudaMemset(deviceCount, 0, sizeof(uint32_t));
        compactPattern<<<1, 64>>>(deviceOutput, deviceCount, capacity, 0);
        cudaDeviceSynchronize();

        std::vector<uint32_t> output(capacity);
        uint32_t count = 0;
        cudaMemcpy(output.data(), deviceOutput, sizeof(uint32_t) * output.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count, deviceCount, sizeof(count), cudaMemcpyDeviceToHost);
        valid &= validateCompacted(output, count, capacity, 0);
        cudaFree(deviceOutput);
        cudaFree(deviceCount);
    }

    {
        constexpr uint32_t capacity = 7;
        constexpr uint32_t sentinel = 0xcdcdcdcd;
        uint32_t* deviceOutput = nullptr;
        uint32_t* deviceCount = nullptr;
        cudaMalloc(&deviceOutput, sizeof(uint32_t) * (capacity + 4));
        cudaMalloc(&deviceCount, sizeof(uint32_t));
        cudaMemset(deviceOutput, 0xcd, sizeof(uint32_t) * (capacity + 4));
        cudaMemset(deviceCount, 0, sizeof(uint32_t));
        compactPattern<<<1, 64>>>(deviceOutput, deviceCount, capacity, 1);
        cudaDeviceSynchronize();

        std::vector<uint32_t> output(capacity + 4);
        uint32_t count = 0;
        cudaMemcpy(output.data(), deviceOutput, sizeof(uint32_t) * output.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count, deviceCount, sizeof(count), cudaMemcpyDeviceToHost);
        valid &= validateCompacted(output, count, capacity, 1);
        for (uint32_t i = capacity; i < output.size(); ++i)
            valid &= check(output[i] == sentinel, "bounded compaction wrote past queue capacity");
        cudaFree(deviceOutput);
        cudaFree(deviceCount);
    }

    return valid ? 0 : 1;
}
