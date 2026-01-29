#ifndef CLAHE_H
#define CLAHE_H

#include <cstdint>
#include <memory>

// CLAHE (Contrast Limited Adaptive Histogram Equalization)
// Mejora el contraste local de forma adaptativa, evitando amplificar demasiado el ruido
//
// Parámetros:
//   - imagen: imagen en escala de grises (uint8_t)
//   - ancho, alto: dimensiones de la imagen
//   - tilesX, tilesY: número de tiles en X e Y (recomendado: 8x8)
//   - clipLimit: límite para recortar el histograma (evita sobre-amplificación, recomendado: 2.0-4.0)
//
// Retorna: nueva imagen procesada con CLAHE
std::unique_ptr<uint8_t[]> aplicarCLAHE(const uint8_t* imagen, int ancho, int alto,
                                         int tilesX, int tilesY, double clipLimit);

#endif
