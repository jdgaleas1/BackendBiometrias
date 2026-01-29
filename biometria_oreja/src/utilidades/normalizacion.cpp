#include "utilidades/normalizacion.h"
#include <numeric>
#include <iostream>
#include <cmath>

std::vector<double> normalizarHistograma(const std::vector<int>& histograma, bool debug) {
    std::vector<double> resultado(histograma.size(), 0.0);
    double suma = std::accumulate(histograma.begin(), histograma.end(), 0.0);

    if (suma <= 1e-10) {
        if (debug)
            std::cout << "Histograma vacio o nulo. Retornando ceros.\n";
        return resultado;
    }

    for (size_t i = 0; i < histograma.size(); ++i)
        resultado[i] = static_cast<double>(histograma[i]) / suma;

    if (debug) {
        double total = std::accumulate(resultado.begin(), resultado.end(), 0.0);
        std::cout << "Suma de vector normalizado: " << total << "\n";
    }

    return resultado;
}

void normalizarVector(std::vector<double>& vec) {
    double norma = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    if (norma > 1e-8) {
        for (auto& val : vec)
            val /= norma;
    }
}

