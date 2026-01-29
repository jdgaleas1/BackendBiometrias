// Archivo: mejoras_preprocesamiento.h
#ifndef MEJORAS_PREPROCESAMIENTO_H
#define MEJORAS_PREPROCESAMIENTO_H

#include <memory>
#include <cstdint>

// CLAHE local sobre regi�n de oreja
std::unique_ptr<uint8_t[]> aplicarCLAHELocal(const uint8_t* imagen, const uint8_t* mascara, int ancho, int alto, int tileSize);

// Recorte autom�tico por bounding box en base a m�scara
std::unique_ptr<uint8_t[]> recortarBoundingBox(const uint8_t* imagen, const uint8_t* mascara, int ancho, int alto, int& outAncho, int& outAlto, int padding);

std::unique_ptr<uint8_t[]> recortarBoundingBoxMascara(const uint8_t* mascara, int ancho, int alto, int& outAncho, int& outAlto, int padding);

std::unique_ptr<uint8_t[]> redimensionarMascaraSimple(const uint8_t* imagen, int anchoOrig, int altoOrig, int anchoObj, int altoObj);

// Crea una máscara elíptica fija y consistente (FASE 1 - Solución 1A)
std::unique_ptr<uint8_t[]> crearMascaraElipticaFija(int ancho, int alto);

// FASE 4 - Filtro Gaussiano consistente para reducir ruido de alta frecuencia
std::unique_ptr<uint8_t[]> aplicarFiltroGaussiano(const uint8_t* imagen, int ancho, int alto, double sigma);

#endif
