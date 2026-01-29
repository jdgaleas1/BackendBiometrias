#ifndef BILATERAL_FILTER_H
#define BILATERAL_FILTER_H

#include <cstdint>
#include <memory>

// Bilateral Filter (filtro bilateral)
// Reduce ruido preservando bordes, ideal para preprocesar antes de extraer features
//
// Parámetros:
//   - imagen: imagen en escala de grises (uint8_t)
//   - ancho, alto: dimensiones de la imagen
//   - sigmaSpace: desviación estándar espacial (controla el tamaño del kernel, recomendado: 3-5)
//   - sigmaColor: desviación estándar de color (controla cuánto se preservan bordes, recomendado: 50-75)
//
// Retorna: nueva imagen filtrada
std::unique_ptr<uint8_t[]> aplicarBilateral(const uint8_t* imagen, int ancho, int alto,
                                             double sigmaSpace, double sigmaColor);

#endif
