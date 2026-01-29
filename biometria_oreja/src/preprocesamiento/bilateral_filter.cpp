#include "preprocesamiento/bilateral_filter.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace {
    // Pre-calcular tabla de Gaussiana espacial
    void calcularKernelEspacial(double sigmaSpace, int radio, double* kernel) {
        double factor = -0.5 / (sigmaSpace * sigmaSpace);
        for (int dy = -radio; dy <= radio; ++dy) {
            for (int dx = -radio; dx <= radio; ++dx) {
                double dist2 = dx * dx + dy * dy;
                int idx = (dy + radio) * (2 * radio + 1) + (dx + radio);
                kernel[idx] = std::exp(factor * dist2);
            }
        }
    }

    // Pre-calcular tabla de pesos de color (evita std::exp en loop interno)
    // Diferencias posibles: 0 a 255
    void calcularTablaPesosColor(double sigmaColor, double* tabla) {
        double factor = -0.5 / (sigmaColor * sigmaColor);
        for (int d = 0; d < 256; ++d) {
            tabla[d] = std::exp(factor * d * d);
        }
    }
}

std::unique_ptr<uint8_t[]> aplicarBilateral(const uint8_t* imagen, int ancho, int alto,
                                             double sigmaSpace, double sigmaColor) {
    if (!imagen || ancho <= 0 || alto <= 0 || sigmaSpace <= 0 || sigmaColor <= 0) {
        return nullptr;
    }

    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    // Radio del kernel (típicamente 2-3 veces sigma)
    int radio = static_cast<int>(std::ceil(3.0 * sigmaSpace));
    if (radio < 1) radio = 1;

    // Pre-calcular kernel espacial
    int kernelSize = (2 * radio + 1) * (2 * radio + 1);
    auto kernelEspacial = std::make_unique<double[]>(kernelSize);
    calcularKernelEspacial(sigmaSpace, radio, kernelEspacial.get());

    // Pre-calcular tabla de pesos de color (evita std::exp en loop interno)
    double tablaPesosColor[256];
    calcularTablaPesosColor(sigmaColor, tablaPesosColor);

    const double* kEsp = kernelEspacial.get();
    const int kernelWidth = 2 * radio + 1;

    // Aplicar filtro bilateral con OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            int centerIdx = y * ancho + x;
            int centerVal = imagen[centerIdx];

            double suma = 0.0;
            double pesoTotal = 0.0;

            // Iterar sobre la ventana
            for (int dy = -radio; dy <= radio; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= alto) continue;

                for (int dx = -radio; dx <= radio; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= ancho) continue;

                    int neighborIdx = ny * ancho + nx;
                    int neighborVal = imagen[neighborIdx];

                    // Peso espacial (pre-calculado)
                    int kernelIdx = (dy + radio) * kernelWidth + (dx + radio);
                    double wSpace = kEsp[kernelIdx];

                    // Peso de color (tabla pre-calculada, sin std::exp)
                    int diff = std::abs(centerVal - neighborVal);
                    double wColor = tablaPesosColor[diff];

                    // Peso total = espacial × color
                    double peso = wSpace * wColor;

                    suma += neighborVal * peso;
                    pesoTotal += peso;
                }
            }

            // Normalizar y guardar
            if (pesoTotal > 0) {
                double resultado = suma / pesoTotal;
                salida[centerIdx] = static_cast<uint8_t>(std::clamp(std::round(resultado), 0.0, 255.0));
            } else {
                salida[centerIdx] = static_cast<uint8_t>(centerVal);
            }
        }
    }

    return salida;
}
