#include "preprocesamiento/mejoras_preprocesamiento.h"
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>

// Aplica CLAHE en una region con mascara binaria (1=oreja)
std::unique_ptr<uint8_t[]> aplicarCLAHELocal(const uint8_t* imagen, const uint8_t* mascara, int ancho, int alto, int tileSize) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);
    std::memcpy(salida.get(), imagen, ancho * alto);

    const int nTilesX = tileSize > 0 ? ancho / tileSize : 0;
    const int nTilesY = tileSize > 0 ? alto / tileSize : 0;

    for (int ty = 0; ty < nTilesY; ++ty) {
        for (int tx = 0; tx < nTilesX; ++tx) {
            std::vector<int> hist(256, 0);
            int x0 = tx * tileSize;
            int y0 = ty * tileSize;

            // Construir histograma local solo en region oreja
            for (int y = y0; y < y0 + tileSize && y < alto; ++y)
                for (int x = x0; x < x0 + tileSize && x < ancho; ++x)
                    if (mascara[y * ancho + x] == 255)
                        hist[imagen[y * ancho + x]]++;

            // Limitar contraste
            int limit = tileSize * tileSize / 4;
            int exceso = 0;
            for (int& h : hist) {
                if (h > limit) {
                    exceso += h - limit;
                    h = limit;
                }
            }
            int redistribucion = exceso / 256;
            for (int& h : hist) h += redistribucion;

            std::vector<int> cdf(256, 0);
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + hist[i];
            int total = cdf[255];

            if (total <= 0) continue;

            for (int y = y0; y < y0 + tileSize && y < alto; ++y) {
                for (int x = x0; x < x0 + tileSize && x < ancho; ++x) {
                    if (mascara[y * ancho + x] == 255) {
                        uint8_t val = imagen[y * ancho + x];
                        salida[y * ancho + x] = static_cast<uint8_t>((cdf[val] * 255) / total);
                    }
                }
            }
        }
    }
    return salida;
}

std::unique_ptr<uint8_t[]> recortarBoundingBox(const uint8_t* imagen, const uint8_t* mascara, int ancho, int alto,
    int& outAncho, int& outAlto, int padding) {
    int minX = ancho, maxX = -1, minY = alto, maxY = -1;
    for (int y = 0; y < alto; ++y)
        for (int x = 0; x < ancho; ++x)
            if (mascara[y * ancho + x] == 255) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }

    if (maxX < 0) return nullptr;

    minX = std::max(0, minX - padding);
    maxX = std::min(ancho - 1, maxX + padding);
    minY = std::max(0, minY - padding);
    maxY = std::min(alto - 1, maxY + padding);

    outAncho = maxX - minX + 1;
    outAlto = maxY - minY + 1;

    auto salida = std::make_unique<uint8_t[]>(outAncho * outAlto);
    for (int y = 0; y < outAlto; ++y)
        for (int x = 0; x < outAncho; ++x)
            salida[y * outAncho + x] = imagen[(minY + y) * ancho + (minX + x)];

    return salida;
}

std::unique_ptr<uint8_t[]> recortarBoundingBoxMascara(const uint8_t* mascara, int ancho, int alto, int& outAncho, int& outAlto, int padding) {
    int minX = ancho, maxX = -1, minY = alto, maxY = -1;
    for (int y = 0; y < alto; ++y)
        for (int x = 0; x < ancho; ++x)
            if (mascara[y * ancho + x] == 255) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }

    if (maxX < 0) return nullptr;

    minX = std::max(0, minX - padding);
    maxX = std::min(ancho - 1, maxX + padding);
    minY = std::max(0, minY - padding);
    maxY = std::min(alto - 1, maxY + padding);

    outAncho = maxX - minX + 1;
    outAlto = maxY - minY + 1;

    auto salida = std::make_unique<uint8_t[]>(outAncho * outAlto);
    for (int y = 0; y < outAlto; ++y)
        for (int x = 0; x < outAncho; ++x)
            salida[y * outAncho + x] = mascara[(minY + y) * ancho + (minX + x)];

    return salida;
}

std::unique_ptr<uint8_t[]> redimensionarMascaraSimple(const uint8_t* imagen, int anchoOrig, int altoOrig, int anchoObj, int altoObj) {
    auto salida = std::make_unique<uint8_t[]>(anchoObj * altoObj);

    float scaleX = static_cast<float>(anchoOrig) / anchoObj;
    float scaleY = static_cast<float>(altoOrig) / altoObj;

    for (int y = 0; y < altoObj; ++y) {
        for (int x = 0; x < anchoObj; ++x) {
            int srcX = std::clamp(static_cast<int>(x * scaleX), 0, anchoOrig - 1);
            int srcY = std::clamp(static_cast<int>(y * scaleY), 0, altoOrig - 1);
            salida[y * anchoObj + x] = imagen[srcY * anchoOrig + srcX];
        }
    }
    return salida;
}

// ============================================================================
// FASE 1 - Solución 1A: Máscara elíptica FIJA y CONSISTENTE
// ============================================================================
// Esta función crea una máscara elíptica centrada que es IDÉNTICA para todas
// las imágenes. Esto elimina la variabilidad causada por detectarRegionOreja
// que usaba gradientes adaptativos y generaba máscaras inconsistentes.
//
// Parámetros para imagen 128x128:
// - Centro: (64, 64)
// - Radio horizontal: 48 píxeles (75% del ancho)
// - Radio vertical: 56 píxeles (87.5% del alto)
// ============================================================================
std::unique_ptr<uint8_t[]> crearMascaraElipticaFija(int ancho, int alto) {
    auto mascara = std::make_unique<uint8_t[]>(ancho * alto);
    std::fill(mascara.get(), mascara.get() + ancho * alto, 0);

    // Centro de la elipse
    float cx = ancho * 0.5f;
    float cy = alto * 0.5f;

    // Radios de la elipse (75% horizontal, 87.5% vertical)
    float rx = ancho * 0.375f;  // 75% del semi-ancho
    float ry = alto * 0.4375f;  // 87.5% del semi-alto

    // Ecuación de la elipse: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            float dx = (x - cx) / rx;
            float dy = (y - cy) / ry;
            float distancia_normalizada = dx * dx + dy * dy;

            if (distancia_normalizada <= 1.0f) {
                mascara[y * ancho + x] = 255;
            }
        }
    }

    return mascara;
}

// ============================================================================
// FASE 4 - Filtro Gaussiano para reducción de ruido
// ============================================================================
// Aplica un filtro Gaussiano consistente (no adaptativo) a la imagen.
// A diferencia del bilateral, este filtro aplica el mismo kernel a todos los
// píxeles, garantizando consistencia entre imágenes.
//
// Parámetros:
// - sigma: desviación estándar del kernel gaussiano
//   - sigma=0.8-1.0: suavizado ligero (recomendado para preservar detalles)
//   - sigma=1.5-2.0: suavizado moderado
//   - Tamaño kernel: 2*ceil(3*sigma)+1 (99.7% de la distribución)
// ============================================================================
std::unique_ptr<uint8_t[]> aplicarFiltroGaussiano(const uint8_t* imagen, int ancho, int alto, double sigma) {
    if (!imagen || ancho <= 0 || alto <= 0 || sigma <= 0) return nullptr;

    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    // Calcular tamaño del kernel (3 sigma cubre 99.7% de la distribución)
    int radio = static_cast<int>(std::ceil(3.0 * sigma));
    int ksize = 2 * radio + 1;

    // Generar kernel Gaussiano 1D
    std::vector<double> kernel(ksize);
    double suma = 0.0;
    double factor = -1.0 / (2.0 * sigma * sigma);

    for (int i = 0; i < ksize; ++i) {
        int x = i - radio;
        kernel[i] = std::exp(factor * x * x);
        suma += kernel[i];
    }

    // Normalizar kernel
    for (double& k : kernel) {
        k /= suma;
    }

    // Buffer temporal para separabilidad (Gaussiano 2D = Gaussiano 1D horizontal + vertical)
    auto temp = std::make_unique<double[]>(ancho * alto);

    // Paso 1: Convolución horizontal
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            double sum = 0.0;
            for (int k = 0; k < ksize; ++k) {
                int px = std::clamp(x + k - radio, 0, ancho - 1);
                sum += imagen[y * ancho + px] * kernel[k];
            }
            temp[y * ancho + x] = sum;
        }
    }

    // Paso 2: Convolución vertical
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            double sum = 0.0;
            for (int k = 0; k < ksize; ++k) {
                int py = std::clamp(y + k - radio, 0, alto - 1);
                sum += temp[py * ancho + x] * kernel[k];
            }
            salida[y * ancho + x] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
        }
    }

    return salida;
}
