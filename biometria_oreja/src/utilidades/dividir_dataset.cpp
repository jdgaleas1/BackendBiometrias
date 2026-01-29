#include "utilidades/dividir_dataset.h"
#include <map>
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>

void dividirEstratificadoRatio(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    std::vector<std::vector<double>>& X_train, std::vector<int>& y_train,
    std::vector<std::vector<double>>& X_test,  std::vector<int>& y_test,
    double test_ratio,
    int seed
) {
    X_train.clear(); y_train.clear();
    X_test.clear();  y_test.clear();

    std::map<int, std::vector<int>> idxPorClase;
    for (int i = 0; i < (int)y.size(); ++i) {
        idxPorClase[y[i]].push_back(i);
    }

    std::mt19937 gen(seed);

    int totalTrain = 0, totalTest = 0;

    for (auto& [clase, idxs] : idxPorClase) {
        std::shuffle(idxs.begin(), idxs.end(), gen);

        int n = (int)idxs.size();
        if (n < 2) {
            std::cerr << "Clase " << clase << " tiene muy pocas muestras (" << n << "), se deja toda en train.\n";
            for (int id : idxs) { X_train.push_back(X[id]); y_train.push_back(clase); }
            continue;
        }

        int n_test = (int)std::round(n * test_ratio);
        if (n_test < 1) n_test = 1;
        if (n_test > n - 1) n_test = n - 1;

        int n_train = n - n_test;

        for (int i = 0; i < n_train; ++i) {
            int id = idxs[i];
            X_train.push_back(X[id]);
            y_train.push_back(clase);
        }
        for (int i = n_train; i < n; ++i) {
            int id = idxs[i];
            X_test.push_back(X[id]);
            y_test.push_back(clase);
        }

        totalTrain += n_train;
        totalTest  += n_test;
    }

    std::cout << "[Split] Train=" << totalTrain << " Test=" << totalTest
              << " (test_ratio=" << test_ratio << ", seed=" << seed << ")\n";
}
