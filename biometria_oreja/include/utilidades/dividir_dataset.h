#ifndef DIVIDIR_DATASET_H
#define DIVIDIR_DATASET_H

#include <vector>

// Divide el dataset en subconjuntos de entrenamiento y prueba estratificados
void dividirEstratificadoRatio(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    std::vector<std::vector<double>>& X_train, std::vector<int>& y_train,
    std::vector<std::vector<double>>& X_test,  std::vector<int>& y_test,
    double test_ratio,
    int seed
);

#endif
