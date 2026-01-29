#ifndef NORMALIZACION_H
#define NORMALIZACION_H

#include <vector>

// Normaliza un histograma entero a un vector de proporciones en [0, 1]
std::vector<double> normalizarHistograma(const std::vector<int>& histograma, bool debug = false);

void normalizarVector(std::vector<double>& vec);

#endif
