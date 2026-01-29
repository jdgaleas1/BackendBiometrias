#ifndef SVM_ENTRENAMIENTO_H
#define SVM_ENTRENAMIENTO_H

#include <vector>
#include <string>

struct ModeloSVM {
    std::vector<int> clases;
    std::vector<std::vector<double>> pesosPorClase;
    std::vector<double> biasPorClase;
};

// Firma original (orden correcto)
ModeloSVM entrenarSVMOVA(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    double tasaAprendizaje,
    int epocas,
    double C,
    double tolerancia);

void entrenarClasificadorBinarioWarmStart(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& yBin,
    std::vector<double>& w,
    double& b,
    double tasaAprendizaje,
    int epocas,
    double C,
    double tolerancia
);

#endif
