#pragma once
#include <cstdint>
#include <memory>
#include <utility>

struct AlignedCrop128 {
    std::unique_ptr<uint8_t[]> img;   // 128*128
    std::unique_ptr<uint8_t[]> mask;  // 128*128 (0/255)
};

// Alinea por PCA sobre la máscara (rotación) + recentrado,
// luego hace un recorte coherente (margen relativo) y reescala a 128x128.
AlignedCrop128 alinearYRecortarOreja128(
    const uint8_t* img128,
    const uint8_t* mask128,
    int w = 128,
    int h = 128,
    float margin_frac = 0.08f,   // 8% margen alrededor del bbox
    bool recenter = true
);
