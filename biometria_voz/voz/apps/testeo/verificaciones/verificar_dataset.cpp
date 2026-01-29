#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <climits> 
#include <cmath>
#include <numeric>
#include <sstream>
#include "../../../core/classification/svm.h"
#include "../../../core/process_dataset/dataset.h"
#include "../../../utils/config.h"
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// Función para calcular estadísticas básicas de un vector
struct Estadisticas {
    AudioSample min, max, media, desviacion;
};

Estadisticas calcularEstadisticas(const std::vector<AudioSample>& datos) {
    Estadisticas stats;
    if (datos.empty()) {
        stats = {0.0, 0.0, 0.0, 0.0};
        return stats;
    }
    
    stats.min = *std::min_element(datos.begin(), datos.end());
    stats.max = *std::max_element(datos.begin(), datos.end());
    
    AudioSample suma = std::accumulate(datos.begin(), datos.end(), 0.0);
    stats.media = suma / datos.size();
    
    AudioSample sumaCuadrados = 0.0;
    for (AudioSample val : datos) {
        AudioSample diff = val - stats.media;
        sumaCuadrados += diff * diff;
    }
    stats.desviacion = std::sqrt(sumaCuadrados / datos.size());
    
    return stats;
}

// Función para verificar distribución del dataset de entrenamiento
void verificarDataset() {
    // Usar rutas de config.h
    std::string trainPath = obtenerRutaDatasetTrain();
    std::string testPath = obtenerRutaDatasetTest();

    std::cout << "=== VERIFICACION DEL DATASET ===" << std::endl;
    std::cout << "Dataset Train: " << trainPath << std::endl;
    std::cout << "Dataset Test:  " << testPath << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Verificar que el archivo existe y obtener tamaño
    std::cout << "\n@ Verificando archivo de entrenamiento..." << std::endl;
    std::ifstream check(trainPath, std::ios::binary | std::ios::ate);
    if (!check.is_open()) {
        std::cout << "% ERROR: No se puede abrir: " << trainPath << std::endl;
        std::cout << "   Verifica que el archivo existe" << std::endl;
        std::cout << "   Ruta absoluta: " << fs::absolute(trainPath) << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }
    
    std::streamsize fileSize = check.tellg();
    std::cout << "  Archivo encontrado" << std::endl;
    std::cout << "  Tamaño: " << (fileSize / 1024.0 / 1024.0) << " MB (" << fileSize << " bytes)" << std::endl;
    check.close();

    std::vector<std::vector<AudioSample>> X;
    std::vector<int> y;

    std::cout << "\n@ Cargando dataset de entrenamiento..." << std::endl;
    if (!cargarDatasetBinario(trainPath, X, y)) {
        std::cout << "% ERROR: Error cargando dataset" << std::endl;
        std::cout << "   El archivo podría estar corrupto o en formato incorrecto" << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }

    // Validaciones básicas
    if (X.empty() || y.empty()) {
        std::cout << "% ERROR: Dataset vacío" << std::endl;
        std::cout << "   X.size() = " << X.size() << ", y.size() = " << y.size() << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }

    if (X.size() != y.size()) {
        std::cout << "% ERROR: Desincronización entre características y etiquetas" << std::endl;
        std::cout << "   X.size() = " << X.size() << ", y.size() = " << y.size() << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }

    // Verificar dimensionalidad consistente
    size_t dimensionEsperada = X[0].size();
    bool dimensionInconsistente = false;
    for (size_t i = 1; i < X.size(); ++i) {
        if (X[i].size() != dimensionEsperada) {
            std::cout << "% ADVERTENCIA: Muestra " << i << " tiene dimensión " << X[i].size() 
                      << " (esperada: " << dimensionEsperada << ")" << std::endl;
            dimensionInconsistente = true;
            if (i > 10) { // Limitar cantidad de warnings
                std::cout << "   ... más inconsistencias detectadas" << std::endl;
                break;
            }
        }
    }

    // Contar muestras por clase
    std::map<int, int> conteoClases;
    for (int label : y) {
        conteoClases[label]++;
    }

    std::cout << "\n@ Dataset cargado exitosamente" << std::endl;
    std::cout << "  Total muestras: " << X.size() << std::endl;
    std::cout << "  Total clases únicas: " << conteoClases.size() << std::endl;
    std::cout << "  Dimensión características: " << dimensionEsperada << std::endl;
    if (dimensionInconsistente) {
        std::cout << "  % ADVERTENCIA: Dimensiones inconsistentes detectadas" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    // Crear vector para ordenar por cantidad de muestras
    std::vector<std::pair<int, int>> clasesOrdenadas;
    for (auto& [clase, count] : conteoClases) {
        clasesOrdenadas.push_back({ count, clase }); // count primero para ordenar
    }

    // Ordenar de mayor a menor cantidad
    std::sort(clasesOrdenadas.begin(), clasesOrdenadas.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Mostrar distribución ordenada
    std::cout << "\n=== DISTRIBUCION POR CLASE (Mayor a Menor) ===" << std::endl;
    std::cout << "Rank | Clase | Muestras | Porcentaje | Gráfico" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    int minMuestras = INT_MAX;
    int maxMuestras = 0;
    int claseDominante = -1;
    int claseMinoritaria = -1;
    std::vector<int> clasesConPocasMuestras; // < 10 muestras

    for (size_t i = 0; i < clasesOrdenadas.size(); ++i) {
        int count = clasesOrdenadas[i].first;
        int clase = clasesOrdenadas[i].second;
        double porcentaje = (100.0 * count) / X.size();

        // Crear gráfico de barras simple
        int barras = static_cast<int>(porcentaje / 2.0); // 1 barra = 2%
        std::string grafico(std::min(barras, 25), '█'); // Máximo 25 caracteres

        std::cout << std::setw(4) << (i + 1) << " | "
            << std::setw(5) << clase << " | "
            << std::setw(8) << count << " | "
            << std::setw(6) << std::fixed << std::setprecision(1) << porcentaje << "% | "
            << grafico;
        
        // Advertencias inline
        if (count < 10) {
            std::cout << " % CRITICO";
            clasesConPocasMuestras.push_back(clase);
        } else if (count < 20) {
            std::cout << " % BAJO";
            clasesConPocasMuestras.push_back(clase);
        }
        
        std::cout << std::endl;

        if (count > maxMuestras) {
            maxMuestras = count;
            claseDominante = clase;
        }
        if (count < minMuestras) {
            minMuestras = count;
            claseMinoritaria = clase;
        }
    }

    double ratio = (double)maxMuestras / minMuestras;

    // Análisis de características
    std::cout << "\n=== ANALISIS DE CARACTERISTICAS ===" << std::endl;
    
    // Detección de problemas en TODAS las características
    int countNaN = 0, countInf = 0, countZero = 0;
    int totalValores = X.size() * dimensionEsperada;
    
    for (const auto& muestra : X) {
        for (AudioSample val : muestra) {
            if (std::isnan(val)) countNaN++;
            else if (std::isinf(val)) countInf++;
            else if (val == 0.0) countZero++;
        }
    }
    
    std::cout << "Total valores: " << totalValores << std::endl;
    std::cout << "Valores cero: " << countZero << " (" 
              << std::fixed << std::setprecision(1) 
              << (100.0f * countZero / totalValores) << "%)" << std::endl;
    
    if (countNaN > 0 || countInf > 0) {
        std::cout << "\n% PROBLEMAS CRITICOS DETECTADOS:" << std::endl;
        if (countNaN > 0) {
            std::cout << "  % NaN encontrados: " << countNaN << " (" 
                      << (100.0f * countNaN / totalValores) << "%)" << std::endl;
            std::cout << "     -> CRITICO: Estos valores romperán el entrenamiento" << std::endl;
            std::cout << "     -> SOLUCION: Reprocesar el dataset eliminando/reemplazando NaN" << std::endl;
        }
        if (countInf > 0) {
            std::cout << "  % Infinitos encontrados: " << countInf << " (" 
                      << (100.0f * countInf / totalValores) << "%)" << std::endl;
            std::cout << "     -> CRITICO: Estos valores romperán el entrenamiento" << std::endl;
            std::cout << "     -> SOLUCION: Reprocesar el dataset eliminando/reemplazando Inf" << std::endl;
        }
    } else {
        std::cout << "  @ No se detectaron NaN ni Inf (dataset válido)" << std::endl;
    }
    
    float porcentajeZeros = (100.0f * countZero) / totalValores;
    if (porcentajeZeros > 80.0f) {
        std::cout << "\n% ADVERTENCIA: " << std::fixed << std::setprecision(1) 
                  << porcentajeZeros << "% de valores son cero (dataset muy disperso)" << std::endl;
        std::cout << "   -> Esto puede indicar características poco informativas" << std::endl;
    } else if (porcentajeZeros > 50.0f) {
        std::cout << "\n@ INFO: " << std::fixed << std::setprecision(1) 
                  << porcentajeZeros << "% de valores son cero (normal para MFCC)" << std::endl;
    }
    
    // Estadísticas de una característica de ejemplo (dim 0)
    std::vector<AudioSample> primeraCaracteristica;
    for (const auto& muestra : X) {
        if (!muestra.empty()) {
            primeraCaracteristica.push_back(muestra[0]);
        }
    }
    
    if (!primeraCaracteristica.empty()) {
        Estadisticas stats = calcularEstadisticas(primeraCaracteristica);
        std::cout << "\nEjemplo (Dim 0):" << std::endl;
        std::cout << "  Rango: [" << stats.min << ", " << stats.max << "]" << std::endl;
        std::cout << "  Media: " << stats.media << ", Desv: " << stats.desviacion << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "=== DIAGNOSTICO DEL DATASET ===" << std::endl;
    std::cout << "\nBALANCE DE CLASES:" << std::endl;
    std::cout << "  Clase dominante: " << claseDominante << " con " << maxMuestras << " muestras" << std::endl;
    std::cout << "  Clase minoritaria: " << claseMinoritaria << " con " << minMuestras << " muestras" << std::endl;
    std::cout << "  Ratio desbalance: " << std::fixed << std::setprecision(1) << ratio << ":1" << std::endl;
    std::cout << "  Promedio muestras/clase: " << (X.size() / conteoClases.size()) << std::endl;
    
    if (!clasesConPocasMuestras.empty()) {
        std::cout << "\n% CLASES CON POCAS MUESTRAS (< 20):" << std::endl;
        std::cout << "  Total: " << clasesConPocasMuestras.size() << " clases" << std::endl;
        std::cout << "  IDs: ";
        for (size_t i = 0; i < std::min(clasesConPocasMuestras.size(), size_t(10)); ++i) {
            std::cout << clasesConPocasMuestras[i];
            if (i < std::min(clasesConPocasMuestras.size(), size_t(10)) - 1) std::cout << ", ";
        }
        if (clasesConPocasMuestras.size() > 10) {
            std::cout << " ... (+" << (clasesConPocasMuestras.size() - 10) << " más)";
        }
        std::cout << std::endl;
    }

    // Recomendaciones basadas en el ratio
    std::cout << "\n-> RECOMENDACIONES:" << std::endl;
    
    if (!clasesConPocasMuestras.empty()) {
        std::cout << "   % CRITICO: " << clasesConPocasMuestras.size() << " clases con < 20 muestras" << std::endl;
        std::cout << "      -> Estas clases NO se entrenarán bien" << std::endl;
        std::cout << "      -> SOLUCION: Recolectar más muestras o eliminar esas clases" << std::endl;
        std::cout << std::endl;
    }
    
    if (ratio > 10.0) {
        std::cout << "   % DATASET EXTREMADAMENTE DESBALANCEADO (ratio > 10:1)" << std::endl;
        std::cout << "      -> El modelo tendera a predecir clases dominantes" << std::endl;
        std::cout << "      -> SOLUCIONES:" << std::endl;
        std::cout << "         1. Submuestrear clases dominantes" << std::endl;
        std::cout << "         2. Sobremuestrear clases minoritarias (data augmentation)" << std::endl;
        std::cout << "         3. Usar pesos adaptativos en CONFIG_SVM" << std::endl;
    }
    else if (ratio > 5.0) {
        std::cout << "   @ Dataset moderadamente desbalanceado (ratio > 5:1)" << std::endl;
        std::cout << "      -> Ajustar parametros de regularizacion (C mas alto)" << std::endl;
        std::cout << "      -> Usar pesos adaptativos en el entrenamiento" << std::endl;
    }
    else if (ratio > 2.0) {
        std::cout << "   @ Dataset levemente desbalanceado (ratio > 2:1)" << std::endl;
        std::cout << "      -> Aceptable para entrenamiento, pero monitorear clases minoritarias" << std::endl;
    }
    else {
        std::cout << "   @ Dataset bien balanceado (ratio < 2:1)" << std::endl;
        std::cout << "      -> Balance ideal para entrenamiento SVM" << std::endl;
    }

    std::cout << std::string(70, '=') << std::endl;
}

int main() {
    verificarDataset();

    std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
    std::cin.get();

    return 0;
}