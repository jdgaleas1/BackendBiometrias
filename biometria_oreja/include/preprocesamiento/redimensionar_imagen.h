#ifndef REDIMENSIONAR_IMAGEN_H
#define REDIMENSIONAR_IMAGEN_H

#include <cstdint>
#include <memory>

std::unique_ptr<uint8_t[]> redimensionarParaBiometria(const uint8_t* imagen, int anchoOrig, int altoOrig, int anchoObj = 128, int altoObj = 128);

#endif
