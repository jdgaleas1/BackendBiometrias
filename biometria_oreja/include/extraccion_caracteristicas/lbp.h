#ifndef LBP_H
#define LBP_H

#include <cstdint>
#include <vector>
#include <memory>

std::unique_ptr<uint8_t[]> aplicarLBPConMascara(const uint8_t* imagen, const uint8_t* mascara,
    int ancho, int alto);

std::vector<int> calcularLBPPorBloquesRobusto(const uint8_t* imagen, const uint8_t* mascara,
    int ancho, int alto,
    int bloquesX, int bloquesY,
    int minPixValidosPorBloque = 20,
    bool usarMascara = true);

std::vector<double> calcularLBPPorBloquesRobustoNorm(const uint8_t* imagen, const uint8_t* mascara,
    int ancho, int alto,
    int bloquesX, int bloquesY,
    int minPixValidosPorBloque = 20,
    bool usarMascara = true);

std::vector<double> normalizarLBPPorBloquesRootL2(const std::vector<int>& histPorBloques,
    int bloquesX, int bloquesY);

// FASE 4 - Multi-Scale LBP: Combina radius=1 y radius=2 (118 bins por bloque)
std::vector<double> calcularLBPMultiEscalaPorBloquesRobustoNorm(const uint8_t* imagen, const uint8_t* mascara,
    int ancho, int alto,
    int bloquesX, int bloquesY,
    int minPixValidosPorBloque = 20,
    bool usarMascara = true);

#endif
