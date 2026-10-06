#include "core/rng.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

struct SeedRecord {
    uint64_t state = 0;
    uint32_t pixel = 0;
    uint32_t sample = 0;
};

bool checkPcgSeedUniqueness() {
    constexpr uint32_t width = 64;
    constexpr uint32_t height = 64;
    constexpr uint32_t samples = 256;
    constexpr uint32_t pixelCount = width * height;

    std::vector<SeedRecord> records;
    records.reserve(static_cast<size_t>(pixelCount) * samples);

    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t pixel = 0; pixel < pixelCount; ++pixel) {
            const uint64_t key = pixelRngKey(pixel, sample, RngStrategy::Pcg32);
            const RngState state = makeRngState(key, RngStrategy::Pcg32);
            records.push_back({state.state, pixel, sample});
        }
    }

    std::sort(records.begin(), records.end(), [](const SeedRecord& left, const SeedRecord& right) {
        return left.state < right.state;
    });

    for (size_t index = 1; index < records.size(); ++index) {
        if (records[index - 1].state == records[index].state) {
            std::cerr << "PCG seed collision between pixel " << records[index - 1].pixel
                      << ", sample " << records[index - 1].sample << " and pixel "
                      << records[index].pixel << ", sample " << records[index].sample << '\n';
            return false;
        }
    }

    return true;
}

int main() {
    bool valid = true;

    const uint64_t key = pixelRngKey(1234, 567, RngStrategy::Pcg32);
    const RngState first = makeRngState(key, RngStrategy::Pcg32);
    const RngState second = makeRngState(key, RngStrategy::Pcg32);
    valid &= first.state == second.state && first.strategy == second.strategy;
    valid &= checkPcgSeedUniqueness();

    if (!valid) {
        std::cerr << "RNG tests failed\n";
        return 1;
    }

    std::cout << "RNG tests passed\n";
    return 0;
}
