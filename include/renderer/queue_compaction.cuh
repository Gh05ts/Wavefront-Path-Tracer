#pragma once

#include <cstdint>

// Appends active lanes from the current warp as one contiguous reservation.
// The count may exceed capacity when a caller intentionally uses a bounded
// output queue; the capacity check protects the output array in that case.
template <typename T>
__device__ inline void appendWarpCompacted(
    bool active,
    const T& item,
    T* outputItems,
    uint32_t* outputCount,
    uint32_t outputCapacity) {
    constexpr uint32_t warpSize = 32;
    uint32_t lane = threadIdx.x % warpSize;
    unsigned int warpMask = __activemask();
    unsigned int activeMask = __ballot_sync(warpMask, active);
    uint32_t rank = __popc(activeMask & (lane == 0 ? 0u : ((1u << lane) - 1u)));
    uint32_t activeCount = __popc(activeMask);
    uint32_t warpBase = 0;

    if (lane == 0 && activeCount > 0)
        warpBase = atomicAdd(outputCount, activeCount);
    warpBase = __shfl_sync(warpMask, warpBase, 0);

    if (active && warpBase + rank < outputCapacity)
        outputItems[warpBase + rank] = item;
}
