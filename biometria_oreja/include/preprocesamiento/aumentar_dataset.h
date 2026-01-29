#ifndef AUMENTAR_DATASET_H
#define AUMENTAR_DATASET_H

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

// FASE 4 - Data Augmentation Geométrico (NO fotométrico)
// Genera variaciones geométricas consistentes de una imagen para aumentar dataset
std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>>
aumentarImagenGeometrico(const uint8_t* original, int ancho, int alto, const std::string& nombreBase);

// Transformaciones geométricas individuales
std::unique_ptr<uint8_t[]> rotarImagen(const uint8_t* img, int ancho, int alto, double angulo_grados);
std::unique_ptr<uint8_t[]> trasladarImagen(const uint8_t* img, int ancho, int alto, int dx, int dy);
std::unique_ptr<uint8_t[]> escalarImagen(const uint8_t* img, int ancho, int alto, double factor);
std::unique_ptr<uint8_t[]> flipHorizontal(const uint8_t* img, int ancho, int alto);

// Funciones legacy (mantener compatibilidad)
std::vector<std::pair<std::unique_ptr<uint8_t[]>, std::string>>
aumentarImagenFotometrica(const uint8_t* original, int ancho, int alto, const std::string& nombreBase);

#endif
