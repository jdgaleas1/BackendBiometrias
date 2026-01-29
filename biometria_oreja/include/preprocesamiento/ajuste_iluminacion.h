#ifndef AJUSTE_ILUMINACION_H
#define AJUSTE_ILUMINACION_H

#include <memory>
#include <cstdint>

// Devuelve una máscara binaria
std::unique_ptr<uint8_t[]> detectarRegionOreja(const uint8_t* imagen, int ancho, int alto);

// Ajuste adaptativo de iluminación con detección de región auricular
std::unique_ptr<uint8_t[]> ajusteIluminacionBiometriaV2(const uint8_t* imagen, int ancho, int alto);

#endif
