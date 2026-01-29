#include "svm.h"
#include <iostream>
#include <limits>
#include <cmath>

// ============================================================================
// PREDICCION Y SCORING
// ============================================================================

int predecirHablante(const std::vector<AudioSample>& x, const ModeloSVM& modelo) {
    if (x.size() != static_cast<size_t>(modelo.dimensionCaracteristicas)) {
        std::cerr << "! Error: Dimension del vector de entrada (" << x.size()
            << ") no coincide con la dimension del modelo ("
            << modelo.dimensionCaracteristicas << ")" << std::endl;
        return -1;
    }

    int mejorClase = -1;
    AudioSample mejorScore = -std::numeric_limits<AudioSample>::max();

    // Prediccion lineal: score = w·x + b
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        AudioSample score = modelo.biasPorClase[i] + dotProduct(modelo.pesosPorClase[i], x);

        if (score > mejorScore) {
            mejorScore = score;
            mejorClase = modelo.clases[i];
        }
    }

    return mejorClase;
}

std::vector<AudioSample> obtenerScores(const std::vector<AudioSample>& x,
    const ModeloSVM& modelo) {
    if (x.size() != static_cast<size_t>(modelo.dimensionCaracteristicas)) {
        std::cerr << "! Error: Dimension del vector de entrada (" << x.size()
            << ") no coincide con la dimension del modelo ("
            << modelo.dimensionCaracteristicas << ")" << std::endl;
        return std::vector<AudioSample>();
    }

    std::vector<AudioSample> scores(modelo.clases.size());

    // Prediccion lineal: score = w·x + b
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        scores[i] = modelo.biasPorClase[i] + dotProduct(modelo.pesosPorClase[i], x);
    }

    return scores;
}