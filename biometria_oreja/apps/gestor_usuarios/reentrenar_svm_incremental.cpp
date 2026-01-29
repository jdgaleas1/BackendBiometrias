#include "svm/cargar_csv.h"
#include "utilidades/pca_utils.h"
#include "utilidades/guardar_csv.h"
#include "utilidades/normalizacion.h"

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>

int main() {
    std::vector<std::vector<double>> X_total;
    std::vector<int> y_total;

    std::cout << "📥 Cargando características LBP desde CSV...\n";
    cargarCSV("out/caracteristicas_fusionadas.csv", X_total, y_total, ';');

    if (X_total.empty() || y_total.empty()) {
        std::cerr << "❌ Error: datos vacíos o mal cargados.\n";
        return 1;
    }

    std::cout << "📊 Total muestras: " << X_total.size()
        << ", Dimensiones: " << X_total[0].size() << "\n";

    // Probar distintos valores de componentes PCA
    std::vector<int> componentes = { 35, 40, 45, 50, 55, 60, 65, 70, 95 };
    for (int n : componentes) {
        std::cout << "\n🔧 Generando PCA con " << n << " componentes...\n";

        ModeloPCA modelo = entrenarPCA(X_total, n);
        std::string rutaModelo = "out/modelo_pca_" + std::to_string(n) + ".dat";
        guardarModeloPCA(rutaModelo, modelo);

        auto X_pca = aplicarPCAConModelo(X_total, modelo);

        for (auto& vec : X_pca)
            normalizarVector(vec);

        // Normalizar (opcional si tu pipeline lo requiere)
        for (auto& fila : X_pca) {
            double norma = std::sqrt(std::inner_product(fila.begin(), fila.end(), fila.begin(), 0.0));
            if (norma > 1e-8)
                for (auto& x : fila) x /= norma;
        }

        std::string rutaCSV = "out/caracteristicas_lda_train" + std::to_string(n) + ".csv";
        guardarCSV(rutaCSV, X_pca, y_total);

        std::cout << "✅ PCA " << n << " componentes guardado en: " << rutaCSV << "\n";
    }

    std::cout << "\n🎯 PCA completado para todos los valores especificados.\n";
    return 0;
}
