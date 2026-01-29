#include "preprocesamiento/filtro_bilateral.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <execution>
#include <memory>

#ifdef PREPROCESO_SEQ
#define EXEC_POLICY std::execution::seq
#else
#define EXEC_POLICY std::execution::par
#endif

namespace detail {

    inline uint8_t calcularMediana(const uint8_t* imagen, int ancho, int alto, int x, int y, int radio) {
        int ksize = (2 * radio + 1) * (2 * radio + 1);
        std::vector<uint8_t> ventana;
        ventana.reserve(ksize);

        for (int ky = -radio; ky <= radio; ++ky) {
            for (int kx = -radio; kx <= radio; ++kx) {
                int px = std::clamp(x + kx, 0, ancho - 1);
                int py = std::clamp(y + ky, 0, alto - 1);
                ventana.push_back(imagen[py * ancho + px]);
            }
        }

        std::nth_element(ventana.begin(), ventana.begin() + ventana.size() / 2, ventana.end());
        return ventana[ventana.size() / 2];
    }

}


std::unique_ptr<uint8_t[]> aplicarFiltroMediana(const uint8_t* imagen, int ancho, int alto, int radio) {
    if (!imagen || ancho <= 0 || alto <= 0) return nullptr;

    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    std::for_each(EXEC_POLICY, salida.get(), salida.get() + ancho * alto, [&](uint8_t& pix) {
        int idx = &pix - salida.get();
        int x = idx % ancho;
        int y = idx / ancho;
        pix = detail::calcularMediana(imagen, ancho, alto, x, y, radio);
        });

    return salida;
}


std::unique_ptr<uint8_t[]> aplicarFiltroBilateral(
    const uint8_t* imagen, int ancho, int alto,
    int radio, double sigma_espacial, double sigma_intensidad) {

    if (!imagen || ancho <= 0 || alto <= 0) return nullptr;
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    std::vector<std::vector<double>> kernel_espacial(2 * radio + 1, std::vector<double>(2 * radio + 1));
    double f_esp = -1.0 / (2.0 * sigma_espacial * sigma_espacial);

    for (int j = -radio; j <= radio; ++j)
        for (int i = -radio; i <= radio; ++i)
            kernel_espacial[j + radio][i + radio] = std::exp(f_esp * (i * i + j * j));

    const double f_int = -1.0 / (2.0 * sigma_intensidad * sigma_intensidad);

    std::for_each(std::execution::par, salida.get(), salida.get() + ancho * alto, [&](uint8_t& pix) {
        int idx = &pix - salida.get();
        int x = idx % ancho;
        int y = idx / ancho;
        double suma = 0.0, pesos = 0.0;
        int central = imagen[y * ancho + x];

        for (int j = -radio; j <= radio; ++j) {
            for (int i = -radio; i <= radio; ++i) {
                int px = std::clamp(x + i, 0, ancho - 1);
                int py = std::clamp(y + j, 0, alto - 1);
                int vecino = imagen[py * ancho + px];

                double w_esp = kernel_espacial[j + radio][i + radio];
                double diff = central - vecino;
                double w_int = std::exp(f_int * diff * diff);

                double w = w_esp * w_int;
                suma += vecino * w;
                pesos += w;
            }
        }

        pix = static_cast<uint8_t>(std::clamp(std::round(suma / pesos), 0.0, 255.0));
        });

    return salida;
}


std::unique_ptr<uint8_t[]> preprocesarImagenOreja(const uint8_t* imagen, int ancho, int alto) {
    if (!imagen || ancho <= 0 || alto <= 0) return nullptr;

    auto paso1 = aplicarFiltroMediana(imagen, ancho, alto, 1);
    return aplicarFiltroBilateral(paso1.get(), ancho, alto, 2, 2.0, 15.0);
}
