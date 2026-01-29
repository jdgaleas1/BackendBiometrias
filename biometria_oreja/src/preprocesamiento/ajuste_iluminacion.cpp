#include "preprocesamiento/ajuste_iluminacion.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <execution>

#ifdef PREPROCESO_SEQ
#define EXEC_POLICY std::execution::seq
#else
#define EXEC_POLICY std::execution::par
#endif

namespace detail {

    constexpr float k_COBERTURA_MINIMA = 0.015f;

    std::unique_ptr<uint8_t[]> calcularGradienteBinario(const uint8_t* imagen, int ancho, int alto) {
        int tam = ancho * alto;
        auto mascara = std::make_unique<uint8_t[]>(tam);
        std::fill(mascara.get(), mascara.get() + tam, 0);

        std::vector<float> gradientes(tam, 0.0f);

        std::for_each(EXEC_POLICY, gradientes.begin() + ancho + 1, gradientes.end() - ancho - 1, [&](float& grad) {
            int idx = &grad - gradientes.data();
            int y = idx / ancho, x = idx % ancho;
            if (x < 1 || x >= ancho - 1 || y < 1 || y >= alto - 1) return;

            int gx = -imagen[(y - 1) * ancho + (x - 1)] + imagen[(y - 1) * ancho + (x + 1)]
                - 2 * imagen[y * ancho + (x - 1)] + 2 * imagen[y * ancho + (x + 1)]
                - imagen[(y + 1) * ancho + (x - 1)] + imagen[(y + 1) * ancho + (x + 1)];

            int gy = -imagen[(y - 1) * ancho + (x - 1)] - 2 * imagen[(y - 1) * ancho + x] - imagen[(y - 1) * ancho + (x + 1)]
                + imagen[(y + 1) * ancho + (x - 1)] + 2 * imagen[(y + 1) * ancho + x] + imagen[(y + 1) * ancho + (x + 1)];

            grad = std::sqrt(gx * gx + gy * gy);
            });

        std::vector<float> copia = gradientes;
        std::nth_element(copia.begin(), copia.begin() + tam * 92 / 100, copia.end());
        float umbral = copia[tam * 92 / 100] * 0.8f;

        for (int i = 0; i < tam; ++i)
            mascara[i] = (gradientes[i] > umbral) ? 1 : 0;

        return mascara;
    }

    void apertura3x3(uint8_t* data, int ancho, int alto) {
        auto tmp = std::make_unique<uint8_t[]>(ancho * alto);

        for (int y = 1; y < alto - 1; ++y)
            for (int x = 1; x < ancho - 1; ++x) {
                int suma = 0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                        suma += data[(y + j) * ancho + (x + i)];
                tmp[y * ancho + x] = (suma >= 7) ? 1 : 0;
            }

        for (int y = 1; y < alto - 1; ++y)
            for (int x = 1; x < ancho - 1; ++x) {
                int suma = 0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                        suma += tmp[(y + j) * ancho + (x + i)];
                data[y * ancho + x] = (suma >= 1) ? 1 : 0;
            }
    }

}

std::unique_ptr<uint8_t[]> detectarRegionOreja(const uint8_t* imagen, int ancho, int alto) {
    int tam = ancho * alto;
    auto mascara = detail::calcularGradienteBinario(imagen, ancho, alto);
    detail::apertura3x3(mascara.get(), ancho, alto);

    auto salida = std::make_unique<uint8_t[]>(tam);
    for (int i = 0; i < tam; ++i)
        salida[i] = mascara[i] ? 255 : 0;

    return salida;
}

std::unique_ptr<uint8_t[]> ajusteIluminacionBiometriaV2(const uint8_t* imagen, int ancho, int alto) {
    if (!imagen || ancho <= 0 || alto <= 0) return nullptr;

    int tam = ancho * alto;
    auto salida = std::make_unique<uint8_t[]>(tam);
    auto mascara = detectarRegionOreja(imagen, ancho, alto);

    int conteoOreja = std::count(mascara.get(), mascara.get() + tam, 255);
    if (static_cast<float>(conteoOreja) / tam < detail::k_COBERTURA_MINIMA) {
        std::copy(imagen, imagen + tam, salida.get());
        return salida;
    }

    uint64_t sumO = 0, sum2O = 0, sumF = 0;
    int pixO = 0, pixF = 0;

    for (int i = 0; i < tam; ++i) {
        uint8_t val = imagen[i];
        if (mascara[i] == 255) {
            sumO += val;
            sum2O += val * val;
            ++pixO;
        }
        else {
            sumF += val;
            ++pixF;
        }
    }

    double mediaO = pixO ? static_cast<double>(sumO) / pixO : 128.0;
    double desvO = pixO ? std::sqrt((static_cast<double>(sum2O) / pixO) - mediaO * mediaO) : 0.0;
    double factor = (mediaO < 100.0) ? 1.2 : (mediaO > 180.0 ? 0.8 : 1.0);
    double gamma = (desvO < 20.0) ? 1.1 : 1.0;
    double mediaF = pixF ? static_cast<double>(sumF) / pixF : 128.0;

    std::for_each(EXEC_POLICY, salida.get(), salida.get() + tam, [&](uint8_t& pix) {
        int i = &pix - salida.get();
        double v = imagen[i] / 255.0;
        double out;

        if (mascara[i] == 255) {
            out = std::pow(v, gamma) * factor;
        }
        else {
            if (mediaF < 50.0) out = v * 1.2;
            else if (mediaF > 200.0) out = v * 0.7;
            else out = v;
        }

        pix = static_cast<uint8_t>(std::clamp(out * 255.0, 0.0, 255.0));
        });

    return salida;
}
