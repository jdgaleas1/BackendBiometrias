#include "svm.h"
#include "../../utils/config.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>
#include <limits>
#include <random>


/**
 * Expande features con terminos cuadraticos para aproximar kernel polinomial grado 2
 *
 * Transforma: [x1, x2, ..., xn] -> [x1, x2, ..., xn, x1�, x2�, ..., xn�]
 *
 * Esto permite al SVM lineal aprender fronteras de decision cuadraticas no lineales,
 * mejorando significativamente la capacidad de separacion del clasificador.
 *
 * @param X Matriz de caracteristicas original (sera modificada in-place)
 */
void expandirFeaturesPolinomial(std::vector<std::vector<AudioSample>>& X) {
    if (X.empty()) return;

    int n_original = static_cast<int>(X[0].size());
    int n_expandido = n_original * 2;  // Features originales + cuadraticas

    std::cout << "-> Expandiendo features polinomiales (grado 2)..." << std::endl;
    std::cout << "   Dimension original: " << n_original << std::endl;
    std::cout << "   Dimension expandida: " << n_expandido
        << " (" << n_original << " + " << n_original << " cuadraticas)" << std::endl;

    for (auto& muestra : X) {
        std::vector<AudioSample> cuadraticas(n_original);

        // Calcular terminos cuadraticos
        for (int i = 0; i < n_original; ++i) {
            cuadraticas[i] = muestra[i] * muestra[i];
        }

        // Concatenar: [originales, cuadraticas]
        muestra.insert(muestra.end(), cuadraticas.begin(), cuadraticas.end());
    }

    std::cout << "   & Expansion completada" << std::endl;
}

// DIAGNOSTICO DE DATASET
void diagnosticarDataset(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y) {

    if (X.empty() || y.empty()) {
        std::cerr << "! Error: Dataset vacio" << std::endl;
        return;
    }

    if (X.size() != y.size()) {
        std::cerr << "! Error: X e y tienen tamanos diferentes" << std::endl;
        return;
    }

    size_t n = X.size();
    size_t dim = X[0].size();

    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   # Muestras totales: " << n << std::endl;
    std::cout << "   # Dimension: " << dim << " caracteristicas" << std::endl;

    // Analizar distribucion de clases
    std::map<int, int> counts;
    for (int label : y) {
        counts[label]++;
    }

    std::cout << "   # Clases detectadas: " << counts.size() << std::endl;
    std::cout << "   # Distribucion por clase:" << std::endl;

    int min_muestras = std::numeric_limits<int>::max();
    int max_muestras = 0;

    for (const auto& [clase, count] : counts) {
        AudioSample porcentaje = 100.0 * count / n;
        std::cout << "      Clase " << std::setw(5) << clase
            << ": " << std::setw(4) << count << " muestras ("
            << std::fixed << std::setprecision(1) << porcentaje << "%)"
            << std::endl;

        min_muestras = std::min(min_muestras, count);
        max_muestras = std::max(max_muestras, count);
    }

    // Calcular ratio de desbalance
    if (min_muestras > 0) {
        AudioSample ratio = static_cast<AudioSample>(max_muestras) / min_muestras;
        std::cout << "   # Ratio desbalance: 1:" << std::fixed << std::setprecision(2)
            << ratio << " (max/min)" << std::endl;

        if (ratio > 5.0) {
            std::cerr << "   % Warning: Dataset muy desbalanceado (ratio > 5.0)"
                << std::endl;
        }
    }

    // Verificar valores invalidos (NaN, Inf)
    int muestras_invalidas = 0;
    int valores_nan = 0;
    int valores_inf = 0;

    for (size_t i = 0; i < X.size(); ++i) {
        bool muestra_valida = true;

        for (AudioSample val : X[i]) {
            if (std::isnan(val)) {
                valores_nan++;
                muestra_valida = false;
            }
            if (std::isinf(val)) {
                valores_inf++;
                muestra_valida = false;
            }
        }

        if (!muestra_valida) {
            muestras_invalidas++;
        }
    }

    if (muestras_invalidas > 0) {
        std::cerr << "   ! ERROR: Valores invalidos detectados!" << std::endl;
        std::cerr << "      Muestras afectadas: " << muestras_invalidas << std::endl;
        std::cerr << "      NaN: " << valores_nan << ", Inf: " << valores_inf << std::endl;
    }
    else {
        std::cout << "   @ Validacion: Todos los valores son numericos validos"
            << std::endl;
    }

    // Estadisticas basicas de las caracteristicas
    std::vector<AudioSample> mins(dim, std::numeric_limits<AudioSample>::max());
    std::vector<AudioSample> maxs(dim, std::numeric_limits<AudioSample>::lowest());
    std::vector<AudioSample> sums(dim, 0.0);

    for (const auto& sample : X) {
        for (size_t j = 0; j < dim; ++j) {
            mins[j] = std::min(mins[j], sample[j]);
            maxs[j] = std::max(maxs[j], sample[j]);
            sums[j] += sample[j];
        }
    }

    // Calcular rangos
    AudioSample rango_min = std::numeric_limits<AudioSample>::max();
    AudioSample rango_max = 0.0;

    for (size_t j = 0; j < dim; ++j) {
        AudioSample rango = maxs[j] - mins[j];
        rango_min = std::min(rango_min, rango);
        rango_max = std::max(rango_max, rango);
    }

    std::cout << "   # Rangos de caracteristicas:" << std::endl;
    std::cout << "      Rango minimo: " << std::fixed << std::setprecision(4)
        << rango_min << std::endl;
    std::cout << "      Rango maximo: " << rango_max << std::endl;

    if (rango_max / (rango_min + 1e-10) > 100.0) {
        std::cerr << "   % Warning: Rangos muy dispares entre caracteristicas"
            << std::endl;
        std::cerr << "      Considera normalizacion o escalado de features" << std::endl;
    }

    std::cout << "   " << std::string(50, '-') << std::endl;
}
