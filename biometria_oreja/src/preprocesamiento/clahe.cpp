#include "preprocesamiento/clahe.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <omp.h>

namespace {
    // Calcular histograma de un tile
    void calcularHistograma(const uint8_t* imagen, int ancho, int alto,
                           int tileX, int tileY, int tileW, int tileH,
                           int hist[256]) {
        std::fill(hist, hist + 256, 0);

        int x0 = tileX * tileW;
        int y0 = tileY * tileH;
        int x1 = std::min(x0 + tileW, ancho);
        int y1 = std::min(y0 + tileH, alto);

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                hist[imagen[y * ancho + x]]++;
            }
        }
    }

    // Recortar histograma (clip) y redistribuir píxeles excedentes
    void recortarHistograma(int hist[256], int clipLimit, int totalPixels) {
        // Recortar bins que exceden el límite
        int exceso = 0;
        for (int i = 0; i < 256; ++i) {
            if (hist[i] > clipLimit) {
                exceso += hist[i] - clipLimit;
                hist[i] = clipLimit;
            }
        }

        // Redistribuir uniformemente el exceso
        int redistribucion = exceso / 256;
        int residuo = exceso % 256;

        for (int i = 0; i < 256; ++i) {
            hist[i] += redistribucion;
        }

        // Distribuir el residuo en los primeros bins
        for (int i = 0; i < residuo; ++i) {
            hist[i]++;
        }
    }

    // Calcular LUT (Look-Up Table) de ecualización para un tile
    void calcularLUT(const int hist[256], uint8_t lut[256], int totalPixels) {
        // Calcular CDF (Cumulative Distribution Function)
        int cdf[256];
        cdf[0] = hist[0];
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        // Normalizar CDF para crear LUT
        int cdfMin = cdf[0];
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] > cdfMin) {
                cdfMin = cdf[i];
                break;
            }
        }

        if (totalPixels <= cdfMin) {
            // No hay suficientes píxeles, LUT identidad
            for (int i = 0; i < 256; ++i) {
                lut[i] = static_cast<uint8_t>(i);
            }
        } else {
            for (int i = 0; i < 256; ++i) {
                int val = ((cdf[i] - cdfMin) * 255) / (totalPixels - cdfMin);
                lut[i] = static_cast<uint8_t>(std::clamp(val, 0, 255));
            }
        }
    }

    // Interpolación bilineal entre 4 LUTs
    uint8_t interpolarBilineal(uint8_t valor,
                               const uint8_t lut1[256], const uint8_t lut2[256],
                               const uint8_t lut3[256], const uint8_t lut4[256],
                               double tx, double ty) {
        // tx, ty ∈ [0, 1]: posición relativa dentro del tile
        double v1 = lut1[valor];
        double v2 = lut2[valor];
        double v3 = lut3[valor];
        double v4 = lut4[valor];

        // Interpolación bilineal
        double top = v1 * (1.0 - tx) + v2 * tx;
        double bot = v3 * (1.0 - tx) + v4 * tx;
        double result = top * (1.0 - ty) + bot * ty;

        return static_cast<uint8_t>(std::clamp(std::round(result), 0.0, 255.0));
    }
}

std::unique_ptr<uint8_t[]> aplicarCLAHE(const uint8_t* imagen, int ancho, int alto,
                                         int tilesX, int tilesY, double clipLimit) {
    if (!imagen || ancho <= 0 || alto <= 0 || tilesX <= 0 || tilesY <= 0) {
        return nullptr;
    }

    // Crear imagen de salida
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);

    // Calcular dimensiones de cada tile
    int tileW = (ancho + tilesX - 1) / tilesX;
    int tileH = (alto + tilesY - 1) / tilesY;

    // Crear LUTs para cada tile
    auto luts = std::make_unique<uint8_t[][256]>(tilesX * tilesY);

    // Procesar cada tile
    for (int tileY_idx = 0; tileY_idx < tilesY; ++tileY_idx) {
        for (int tileX_idx = 0; tileX_idx < tilesX; ++tileX_idx) {
            int hist[256];
            calcularHistograma(imagen, ancho, alto, tileX_idx, tileY_idx, tileW, tileH, hist);

            // Calcular límite de clip en función del número de píxeles
            int x0 = tileX_idx * tileW;
            int y0 = tileY_idx * tileH;
            int x1 = std::min(x0 + tileW, ancho);
            int y1 = std::min(y0 + tileH, alto);
            int totalPixels = (x1 - x0) * (y1 - y0);

            int clipLimitInt = static_cast<int>(clipLimit * totalPixels / 256.0);
            if (clipLimitInt < 1) clipLimitInt = 1;

            recortarHistograma(hist, clipLimitInt, totalPixels);

            int lutIdx = tileY_idx * tilesX + tileX_idx;
            calcularLUT(hist, luts[lutIdx], totalPixels);
        }
    }

    // Aplicar interpolación bilineal entre tiles (paralelizado)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            // Determinar en qué tile estamos
            double fx = (double)x / tileW;
            double fy = (double)y / tileH;

            int tileX_idx = std::min((int)fx, tilesX - 1);
            int tileY_idx = std::min((int)fy, tilesY - 1);

            // Posición relativa dentro del tile [0, 1]
            double tx = fx - tileX_idx;
            double ty = fy - tileY_idx;

            uint8_t valor = imagen[y * ancho + x];

            // Casos especiales: bordes y esquinas (sin interpolación)
            if (tileX_idx == 0 && tileY_idx == 0) {
                // Esquina superior izquierda
                salida[y * ancho + x] = luts[0][valor];
            } else if (tileX_idx == tilesX - 1 && tileY_idx == 0) {
                // Esquina superior derecha
                salida[y * ancho + x] = luts[tilesX - 1][valor];
            } else if (tileX_idx == 0 && tileY_idx == tilesY - 1) {
                // Esquina inferior izquierda
                salida[y * ancho + x] = luts[(tilesY - 1) * tilesX][valor];
            } else if (tileX_idx == tilesX - 1 && tileY_idx == tilesY - 1) {
                // Esquina inferior derecha
                salida[y * ancho + x] = luts[tilesY * tilesX - 1][valor];
            } else if (tileX_idx == 0) {
                // Borde izquierdo (interpolar verticalmente)
                int idx1 = tileY_idx * tilesX;
                int idx2 = (tileY_idx + 1) * tilesX;
                double v1 = luts[idx1][valor];
                double v2 = (tileY_idx + 1 < tilesY) ? luts[idx2][valor] : v1;
                salida[y * ancho + x] = static_cast<uint8_t>(v1 * (1.0 - ty) + v2 * ty);
            } else if (tileX_idx == tilesX - 1) {
                // Borde derecho (interpolar verticalmente)
                int idx1 = tileY_idx * tilesX + tilesX - 1;
                int idx2 = (tileY_idx + 1) * tilesX + tilesX - 1;
                double v1 = luts[idx1][valor];
                double v2 = (tileY_idx + 1 < tilesY) ? luts[idx2][valor] : v1;
                salida[y * ancho + x] = static_cast<uint8_t>(v1 * (1.0 - ty) + v2 * ty);
            } else if (tileY_idx == 0) {
                // Borde superior (interpolar horizontalmente)
                int idx1 = tileX_idx;
                int idx2 = tileX_idx + 1;
                double v1 = luts[idx1][valor];
                double v2 = (tileX_idx + 1 < tilesX) ? luts[idx2][valor] : v1;
                salida[y * ancho + x] = static_cast<uint8_t>(v1 * (1.0 - tx) + v2 * tx);
            } else if (tileY_idx == tilesY - 1) {
                // Borde inferior (interpolar horizontalmente)
                int idx1 = (tilesY - 1) * tilesX + tileX_idx;
                int idx2 = (tilesY - 1) * tilesX + tileX_idx + 1;
                double v1 = luts[idx1][valor];
                double v2 = (tileX_idx + 1 < tilesX) ? luts[idx2][valor] : v1;
                salida[y * ancho + x] = static_cast<uint8_t>(v1 * (1.0 - tx) + v2 * tx);
            } else {
                // Interior: interpolación bilineal entre 4 tiles
                int idx1 = tileY_idx * tilesX + tileX_idx;           // top-left
                int idx2 = tileY_idx * tilesX + tileX_idx + 1;       // top-right
                int idx3 = (tileY_idx + 1) * tilesX + tileX_idx;     // bottom-left
                int idx4 = (tileY_idx + 1) * tilesX + tileX_idx + 1; // bottom-right

                salida[y * ancho + x] = interpolarBilineal(
                    valor, luts[idx1], luts[idx2], luts[idx3], luts[idx4], tx, ty
                );
            }
        }
    }

    return salida;
}
