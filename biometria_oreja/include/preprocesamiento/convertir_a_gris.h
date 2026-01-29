#ifndef CONVERTIR_A_GRIS_H
#define CONVERTIR_A_GRIS_H

#include <cstdint>
#include <memory>

std::unique_ptr<uint8_t[]> convertirAGris(const uint8_t* entradaRGB, int ancho, int alto);

#endif 
