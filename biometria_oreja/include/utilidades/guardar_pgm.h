#ifndef GUARDAR_PGM_H
#define GUARDAR_PGM_H

#include <string>
#include <cstdint>

bool guardarImagenPGM(const std::string& ruta, const uint8_t* imagen, int ancho, int alto);

#endif
