#ifndef FILTRO_BILATERAL_H
#define FILTRO_BILATERAL_H

#include <memory>
#include <cstdint>

// Filtro de mediana
std::unique_ptr<uint8_t[]> aplicarFiltroMediana(const uint8_t* imagen, int ancho, int alto, int radio = 1);

// Filtro bilateral
std::unique_ptr<uint8_t[]> aplicarFiltroBilateral(
    const uint8_t* imagen, int ancho, int alto, int radio = 2,
    double sigma_espacial = 2.0, double sigma_intensidad = 25.0
);

// Filtro completo: primero mediana, luego bilateral
std::unique_ptr<uint8_t[]> preprocesarImagenOreja(
    const uint8_t* imagen, int ancho, int alto
);

#endif
