#include "preprocesamiento/redimensionar_imagen.h"
#include <cmath>
#include <algorithm>
#include <execution>
#include <memory>

#ifdef PREPROCESO_SEQ
#define EXEC_POLICY std::execution::seq
#else
#define EXEC_POLICY std::execution::par
#endif

namespace detail {

    // Interpolación bicúbica
    inline float interpolacionCubica(float p0, float p1, float p2, float p3, float t) {
        float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
        float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
        float c = -0.5f * p0 + 0.5f * p2;
        float d = p1;
        return a * t * t * t + b * t * t + c * t + d;
    }

}

std::unique_ptr<uint8_t[]> redimensionarParaBiometria(const uint8_t* imagen, int anchoOrig, int altoOrig,
    int anchoObj, int altoObj) {
    if (!imagen || anchoOrig <= 0 || altoOrig <= 0 || anchoObj <= 0 || altoObj <= 0) return nullptr;

    auto salida = std::make_unique<uint8_t[]>(anchoObj * altoObj);

    float ratioOrig = static_cast<float>(anchoOrig) / altoOrig;
    float ratioObj = static_cast<float>(anchoObj) / altoObj;
    float diferencia = std::abs(ratioOrig - ratioObj);

    if (diferencia < 0.1f) {
        // Redimensionamiento directo usando interpolación bicúbica
        float escala = static_cast<float>(anchoOrig) / anchoObj;

        std::for_each(EXEC_POLICY, salida.get(), salida.get() + anchoObj * altoObj, [&](uint8_t& px) {
            int idx = &px - salida.get();
            int x = idx % anchoObj;
            int y = idx / anchoObj;

            float gx = (x + 0.5f) * escala - 0.5f;
            float gy = (y + 0.5f) * escala - 0.5f;
            int gxi = static_cast<int>(gx);
            int gyi = static_cast<int>(gy);
            float dx = gx - gxi;
            float dy = gy - gyi;

            float line[4];
            for (int j = 0; j < 4; ++j) {
                int srcY = std::clamp(gyi + j - 1, 0, altoOrig - 1);
                float p0 = imagen[srcY * anchoOrig + std::clamp(gxi - 1, 0, anchoOrig - 1)];
                float p1 = imagen[srcY * anchoOrig + std::clamp(gxi, 0, anchoOrig - 1)];
                float p2 = imagen[srcY * anchoOrig + std::clamp(gxi + 1, 0, anchoOrig - 1)];
                float p3 = imagen[srcY * anchoOrig + std::clamp(gxi + 2, 0, anchoOrig - 1)];
                line[j] = detail::interpolacionCubica(p0, p1, p2, p3, dx);
            }

            float valor = detail::interpolacionCubica(line[0], line[1], line[2], line[3], dy);
            px = static_cast<uint8_t>(std::clamp(valor, 0.0f, 255.0f));
            });

    }
    else {
        // LETTERBOX con interpolación bilineal
        float escala = std::min(static_cast<float>(anchoObj) / anchoOrig, static_cast<float>(altoObj) / altoOrig);
        int nuevoAncho = static_cast<int>(anchoOrig * escala);
        int nuevoAlto = static_cast<int>(altoOrig * escala);
        int offsetX = (anchoObj - nuevoAncho) / 2;
        int offsetY = (altoObj - nuevoAlto) / 2;

        std::fill(salida.get(), salida.get() + anchoObj * altoObj, 0);

        std::for_each(std::execution::par, salida.get(), salida.get() + anchoObj * altoObj, [&](uint8_t& px) {
            int idx = &px - salida.get();
            int x = idx % anchoObj;
            int y = idx / anchoObj;

            int x_img = x - offsetX;
            int y_img = y - offsetY;

            if (x_img < 0 || x_img >= nuevoAncho || y_img < 0 || y_img >= nuevoAlto) {
                px = 0;
                return;
            }

            float gx = (x_img + 0.5f) / escala - 0.5f;
            float gy = (y_img + 0.5f) / escala - 0.5f;
            int gxi = std::clamp(static_cast<int>(gx), 0, anchoOrig - 2);
            int gyi = std::clamp(static_cast<int>(gy), 0, altoOrig - 2);
            float dx = gx - gxi;
            float dy = gy - gyi;

            float interpolado =
                (1 - dx) * (1 - dy) * imagen[gyi * anchoOrig + gxi] +
                dx * (1 - dy) * imagen[gyi * anchoOrig + gxi + 1] +
                (1 - dx) * dy * imagen[(gyi + 1) * anchoOrig + gxi] +
                dx * dy * imagen[(gyi + 1) * anchoOrig + gxi + 1];

            px = static_cast<uint8_t>(std::clamp(interpolado, 0.0f, 255.0f));
            });
    }

    return salida;
}
