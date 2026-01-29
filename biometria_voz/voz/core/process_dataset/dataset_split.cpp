#include "dataset.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <map>
#include <iomanip>

// ============================================================================
// SPLIT TRAIN/TEST ESTRATIFICADO
// ============================================================================

SplitResult dividirTrainTest(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y,
    AudioSample train_ratio,
    unsigned int seed) {

    SplitResult result;

    if (X.empty() || y.empty()) {
        std::cerr << "! Error: Dataset vacio en dividirTrainTest" << std::endl;
        return result;
    }

    if (X.size() != y.size()) {
        std::cerr << "! Error: X e y tienen tamanos diferentes" << std::endl;
        return result;
    }

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        std::cerr << "! Error: train_ratio debe estar en (0, 1)" << std::endl;
        return result;
    }

    std::cout << "\n-> Dividiendo dataset train/test (estratificado)" << std::endl;
    std::cout << "   Ratio train/test: " << std::fixed << std::setprecision(1)
        << (train_ratio * 100) << "% / " << ((1 - train_ratio) * 100) << "%"
        << std::endl;
    std::cout << "   Seed: " << seed << std::endl;

    // Agrupar muestras por clase
    std::map<int, std::vector<size_t>> indices_por_clase;

    for (size_t i = 0; i < y.size(); ++i) {
        indices_por_clase[y[i]].push_back(i);
    }

    std::cout << "   Clases detectadas: " << indices_por_clase.size() << std::endl;

    // Generar seed engine
    std::mt19937 gen(seed);

    // Procesar cada clase
    for (auto& [clase, indices] : indices_por_clase) {
        // Shuffle indices dentro de la clase
        std::shuffle(indices.begin(), indices.end(), gen);

        // Calcular cuántas muestras van a train
        size_t n_train = static_cast<size_t>(indices.size() * train_ratio);

        // Asegurar al menos 1 muestra en cada conjunto
        if (n_train == 0) n_train = 1;
        if (n_train >= indices.size()) n_train = indices.size() - 1;

        size_t n_test = indices.size() - n_train;

        std::cout << "   Clase " << std::setw(5) << clase << ": "
            << std::setw(3) << n_train << " train, "
            << std::setw(3) << n_test << " test "
            << "(de " << indices.size() << " totales)" << std::endl;

        // Dividir en train y test
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = indices[i];

            if (i < n_train) {
                // Train
                result.train.X.push_back(X[idx]);
                result.train.y.push_back(y[idx]);
                result.train_counts[clase]++;
            }
            else {
                // Test
                result.test.X.push_back(X[idx]);
                result.test.y.push_back(y[idx]);
                result.test_counts[clase]++;
            }
        }
    }

    // Validar que todos los conjuntos tengan datos
    if (result.train.empty() || result.test.empty()) {
        std::cerr << "! Error: Train o test vacio despues del split" << std::endl;
        return SplitResult();
    }

    std::cout << "\n   @ Split completado:" << std::endl;
    std::cout << "      Train: " << result.train.size() << " muestras" << std::endl;
    std::cout << "      Test:  " << result.test.size() << " muestras" << std::endl;

    // Verificar que todas las clases esten en ambos conjuntos
    bool todas_en_ambos = true;
    for (const auto& [clase, _] : indices_por_clase) {
        if (result.train_counts[clase] == 0 || result.test_counts[clase] == 0) {
            std::cerr << "   % Warning: Clase " << clase
                << " no tiene muestras en train o test" << std::endl;
            todas_en_ambos = false;
        }
    }

    if (todas_en_ambos) {
        std::cout << "   @ Todas las clases presentes en train y test" << std::endl;
    }

    return result;
}

SplitResult dividirTrainTest(const Dataset& dataset,
    AudioSample train_ratio,
    unsigned int seed) {
    return dividirTrainTest(dataset.X, dataset.y, train_ratio, seed);
}
