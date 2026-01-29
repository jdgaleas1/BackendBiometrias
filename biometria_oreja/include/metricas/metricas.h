#ifndef METRICAS_H
#define METRICAS_H
#include <cstdint>
#include <vector>
#include <utility>

std::pair<double, double> calcularPSNR_SNR(const uint8_t* referencia, const uint8_t* resultado, int ancho, int alto);
double calcularEntropia(const uint8_t* imagen, int ancho, int alto);
double calcularSSIM(const uint8_t* img1, const uint8_t* img2, int ancho, int alto);
double calcularIoU(const uint8_t* prediccion, const uint8_t* referencia, int ancho, int alto);
double calcularDensidadInformacion(const uint8_t* imagen, int ancho, int alto);
double calcularVarianzaExplicada(const std::vector<double>& valoresPropios, int k);

#endif
