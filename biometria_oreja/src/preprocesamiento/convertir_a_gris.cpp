#include "preprocesamiento/convertir_a_gris.h"

#include <cmath>
#include <memory>
#include <algorithm>
#include <execution>

#ifdef PREPROCESO_SEQ
#define EXEC_POLICY std::execution::seq
#else
#define EXEC_POLICY std::execution::par
#endif

namespace { constexpr float WR = 0.35f, WG = 0.45f, WB = 0.20f; }

std::unique_ptr<uint8_t[]> convertirAGris(const uint8_t* entradaRGB, int ancho, int alto) {
    if (!entradaRGB || ancho <= 0 || alto <= 0)
        return nullptr;

    int tam = ancho * alto;
    auto salida = std::make_unique<uint8_t[]>(tam);

    std::for_each(
        EXEC_POLICY,
        salida.get(),
        salida.get() + tam,
        [entradaRGB, salida_ptr = salida.get()](uint8_t& pixel) {
            int idx = &pixel - salida_ptr;
            int r = entradaRGB[idx * 3 + 0];
            int g = entradaRGB[idx * 3 + 1];
            int b = entradaRGB[idx * 3 + 2];
            float gris = WR * r + WG * g + WB * b;
            pixel = static_cast<uint8_t>(std::clamp(std::round(gris), 0.0f, 255.0f));
        }
    );

    return salida;
}
