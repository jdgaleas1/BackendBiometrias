#include "preprocesamiento/aumentar_dataset.h"
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <random>
#include <cstring> 

namespace {

    inline uint8_t clamp(int val) {
        return static_cast<uint8_t>(std::min(255, std::max(0, val)));
    }

    std::unique_ptr<uint8_t[]> ajustarBrillo(const uint8_t* img, int ancho, int alto, int delta) {
        auto salida = std::make_unique<uint8_t[]>(ancho * alto);
        std::transform(img, img + ancho * alto, salida.get(), [delta](uint8_t v) {
            return clamp(v + delta);
            });
        return salida;
    }

    std::unique_ptr<uint8_t[]> ajustarContraste(const uint8_t* img, int ancho, int alto, double factor) {
        auto salida = std::make_unique<uint8_t[]>(ancho * alto);
        std::transform(img, img + ancho * alto, salida.get(), [factor](uint8_t v) {
            return clamp(static_cast<int>((v - 128) * factor + 128));
            });
        return salida;
    }

    std::unique_ptr<uint8_t[]> ajustarGamma(const uint8_t* img, int ancho, int alto, double gamma) {
        uint8_t lut[256];
        const double inv = 1.0 / std::max(1e-6, gamma);
        for (int i = 0; i < 256; ++i) {
            double n = i / 255.0;
            lut[i] = static_cast<uint8_t>(clamp(static_cast<int>(std::round(std::pow(n, inv) * 255.0))));
        }
        auto salida = std::make_unique<uint8_t[]>(ancho * alto);
        for (int i = 0; i < ancho * alto; ++i) salida[i] = lut[img[i]];
        return salida;
    }

    std::unique_ptr<uint8_t[]> agregarRuido(const uint8_t* img, int ancho, int alto, int intensidad) {
        static thread_local std::mt19937 gen{ std::random_device{}() };
        std::uniform_int_distribution<int> dist(-intensidad, intensidad);

        auto salida = std::make_unique<uint8_t[]>(ancho * alto);
        std::transform(img, img + ancho * alto, salida.get(), [&](uint8_t v) {
            return clamp(v + dist(gen));
            });
        return salida;
    }

} 

std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>>
aumentarImagenFotometrica(const uint8_t* original, int ancho, int alto, const std::string& nombreBase) {
    std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>> out;
    out.reserve(6);

    // brillo
    out.emplace_back(ajustarBrillo(original, ancho, alto, 20),  nombreBase + "_b+20");
    out.emplace_back(ajustarBrillo(original, ancho, alto, -15), nombreBase + "_b-15");

    // contraste
    out.emplace_back(ajustarContraste(original, ancho, alto, 1.10), nombreBase + "_c110");

    // gamma
    out.emplace_back(ajustarGamma(original, ancho, alto, 0.90), nombreBase + "_g090");
    out.emplace_back(ajustarGamma(original, ancho, alto, 1.10), nombreBase + "_g110");

    // ruido
    out.emplace_back(agregarRuido(original, ancho, alto, 10), nombreBase + "_n10");

    return out;
}

// ============================================================================
// FASE 4 - Data Augmentation Geométrico
// ============================================================================

// Rotación por ángulo (grados) con interpolación bilineal
std::unique_ptr<uint8_t[]> rotarImagen(const uint8_t* img, int ancho, int alto, double angulo_grados) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);
    std::fill(salida.get(), salida.get() + ancho * alto, 0);

    double angulo_rad = angulo_grados * M_PI / 180.0;
    double cos_a = std::cos(angulo_rad);
    double sin_a = std::sin(angulo_rad);

    int cx = ancho / 2;
    int cy = alto / 2;

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            // Transformación inversa (del destino al origen)
            int dx = x - cx;
            int dy = y - cy;

            double src_x = dx * cos_a + dy * sin_a + cx;
            double src_y = -dx * sin_a + dy * cos_a + cy;

            // Interpolación bilineal
            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));

            if (x0 >= 0 && x0 < ancho - 1 && y0 >= 0 && y0 < alto - 1) {
                double fx = src_x - x0;
                double fy = src_y - y0;

                double v00 = img[y0 * ancho + x0];
                double v10 = img[y0 * ancho + x0 + 1];
                double v01 = img[(y0 + 1) * ancho + x0];
                double v11 = img[(y0 + 1) * ancho + x0 + 1];

                double v = (1 - fx) * (1 - fy) * v00 +
                           fx * (1 - fy) * v10 +
                           (1 - fx) * fy * v01 +
                           fx * fy * v11;

                salida[y * ancho + x] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
            }
        }
    }

    return salida;
}

// Traslación (desplazamiento) en píxeles
std::unique_ptr<uint8_t[]> trasladarImagen(const uint8_t* img, int ancho, int alto, int dx, int dy) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);
    std::fill(salida.get(), salida.get() + ancho * alto, 0);

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            int src_x = x - dx;
            int src_y = y - dy;

            if (src_x >= 0 && src_x < ancho && src_y >= 0 && src_y < alto) {
                salida[y * ancho + x] = img[src_y * ancho + src_x];
            }
        }
    }

    return salida;
}

// Escalado (zoom) por factor. Factor > 1 = zoom in, < 1 = zoom out
std::unique_ptr<uint8_t[]> escalarImagen(const uint8_t* img, int ancho, int alto, double factor) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);
    std::fill(salida.get(), salida.get() + ancho * alto, 0);

    double inv_factor = 1.0 / factor;
    int cx = ancho / 2;
    int cy = alto / 2;

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            // Mapear desde el centro
            double dx = (x - cx) * inv_factor;
            double dy = (y - cy) * inv_factor;

            double src_x = dx + cx;
            double src_y = dy + cy;

            // Interpolación bilineal
            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));

            if (x0 >= 0 && x0 < ancho - 1 && y0 >= 0 && y0 < alto - 1) {
                double fx = src_x - x0;
                double fy = src_y - y0;

                double v00 = img[y0 * ancho + x0];
                double v10 = img[y0 * ancho + x0 + 1];
                double v01 = img[(y0 + 1) * ancho + x0];
                double v11 = img[(y0 + 1) * ancho + x0 + 1];

                double v = (1 - fx) * (1 - fy) * v00 +
                           fx * (1 - fy) * v10 +
                           (1 - fx) * fy * v01 +
                           fx * fy * v11;

                salida[y * ancho + x] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
            }
        }
    }

    return salida;
}

// Flip horizontal (espejo vertical)
std::unique_ptr<uint8_t[]> flipHorizontal(const uint8_t* img, int ancho, int alto) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            salida[y * ancho + x] = img[y * ancho + (ancho - 1 - x)];
        }
    }

    return salida;
}

// ============================================================================
// AUGMENTATION GEOMÉTRICO ALEATORIO (4 variaciones por imagen)
// ============================================================================
// Objetivo: Mejorar generalización con variabilidad real, evitando overfit
// Rangos conservadores para no deformar la identidad:
//   - rotación:  ±6°
//   - traslación: ±2 px
//   - zoom:      0.98–1.02
// ============================================================================
std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>>
aumentarImagenGeometrico(const uint8_t* original, int ancho, int alto, const std::string& nombreBase) {
    std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>> out;
    out.reserve(4);

    static thread_local std::mt19937 gen{ 12345u };
    std::uniform_real_distribution<double> rot_dist(-4.0, 4.0);
    std::uniform_int_distribution<int> shift_dist(-1, 1);
    std::uniform_real_distribution<double> zoom_dist(0.99, 1.01);

    for (int i = 0; i < 4; ++i) {
        const double ang = rot_dist(gen);
        const int dx = shift_dist(gen);
        const int dy = shift_dist(gen);
        const double zoom = zoom_dist(gen);

        auto temp1 = rotarImagen(original, ancho, alto, ang);
        auto temp2 = trasladarImagen(temp1.get(), ancho, alto, dx, dy);
        auto final = escalarImagen(temp2.get(), ancho, alto, zoom);

        out.emplace_back(std::move(final), nombreBase + "_aug" + std::to_string(i + 1));
    }

    return out;
}
