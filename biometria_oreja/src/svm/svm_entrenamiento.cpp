#include "svm/svm_entrenamiento.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

// Entrenamiento OVA con SGD (igual que tenías) + guarda del mejor w,b
ModeloSVM entrenarSVMOVA(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
    double tasaAprendizaje, int epocas, double C, double tolerancia) {
    int m = (int)X.size();
    int n = (int)X[0].size();

    // Clases únicas
    std::vector<int> clases = y;
    std::sort(clases.begin(), clases.end());
    clases.erase(std::unique(clases.begin(), clases.end()), clases.end());

    ModeloSVM modelo;
    modelo.clases = clases;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int c : clases) {
        std::vector<double> w(n, 0.0);
        double b = 0.0;

        // Guardaremos el mejor observado
        std::vector<double> bestW = w;
        double bestB = b;
        double mejorLoss = 1e9;
        int    sinMejora = 0;
        double tasa = tasaAprendizaje;

        for (int epoca = 0; epoca < epocas; ++epoca) {
            // Barajar índices
            std::vector<int> idx(m);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), gen);

            double lossTotal = 0.0;
            double inv_m = 1.0 / m;  // ✅ Calcular una vez, reutilizar

            for (int i : idx) {
                int yi = (y[i] == c) ? 1 : -1;
                double score = std::inner_product(X[i].begin(), X[i].end(), w.begin(), 0.0) + b;
                double margen = yi * score;

                if (margen < 1.0) {
                    // ✅ CORREGIDO: Gradiente de datos dividido por m
                    for (int j = 0; j < n; ++j)
                        w[j] -= tasa * ((-yi * X[i][j] * inv_m) + C * w[j]);  // ✅
                    b -= tasa * (-yi * inv_m);  // ✅
                    lossTotal += (1.0 - margen);
                }
                else {
                    // Solo regularización (no necesita cambio)
                    for (int j = 0; j < n; ++j)
                        w[j] -= tasa * (C * w[j]);
                }
            }
            
            // Resto del código sigue igual...
            double loss = lossTotal / m;

            if (loss < mejorLoss - tolerancia) {
                mejorLoss = loss;
                sinMejora = 0;
                bestW = w;
                bestB = b;
            }
            else {
                ++sinMejora;
            }

            if (epoca % 500 == 0 && epoca > 0)
                tasa *= 0.9;

            if (sinMejora > 300 && epoca > 600) 
                break;
        }

        // Guardar el MEJOR modelo observado
        modelo.pesosPorClase.push_back(bestW);
        modelo.biasPorClase.push_back(bestB);
    }

    return modelo;
}

void entrenarClasificadorBinarioWarmStart(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& yBin,   // +1 o -1
    std::vector<double>& w,         // in/out
    double& b,                      // in/out
    double tasaAprendizaje,
    int epocas,
    double C,
    double tolerancia
) {
    int m = (int)X.size();
    int n = (int)X[0].size();

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> bestW = w;
    double bestB = b;
    double mejorLoss = 1e9;
    int sinMejora = 0;
    double tasa = tasaAprendizaje;

    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);

    for (int epoca = 0; epoca < epocas; ++epoca) {
        std::shuffle(idx.begin(), idx.end(), gen);

        double lossTotal = 0.0;
        double inv_m = 1.0 / m;  // ✅ Calcular una vez, reutilizar

        for (int ii : idx) {
            int yi = yBin[ii]; // +1 o -1
            double score = std::inner_product(X[ii].begin(), X[ii].end(), w.begin(), 0.0) + b;
            double margen = yi * score;

            if (margen < 1.0) {
                // ✅ CORREGIDO: Gradiente de datos dividido por m
                for (int j = 0; j < n; ++j)
                    w[j] -= tasa * ((-yi * X[ii][j] * inv_m) + C * w[j]);  // ✅
                b -= tasa * (-yi * inv_m);  // ✅
                lossTotal += (1.0 - margen);
            }
            else {
                for (int j = 0; j < n; ++j)
                    w[j] -= tasa * (C * w[j]);
            }
        }

        double loss = lossTotal / m;

        if (loss < mejorLoss - tolerancia) {
            mejorLoss = loss;
            sinMejora = 0;
            bestW = w;
            bestB = b;
        }
        else {
            ++sinMejora;
        }

        if (epoca % 200 == 0 && epoca > 0) tasa *= 0.9;

        if (sinMejora > 50 && epoca > 100) break;
    }

    w = std::move(bestW);
    b = bestB;
}
