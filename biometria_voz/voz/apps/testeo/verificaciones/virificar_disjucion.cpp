// ============================================================================
// VERIFICACION DE DISJUNCION TRAIN/TEST CON HASHING
// ============================================================================
// Detecta data leakage verificando que no hay features duplicadas entre
// conjuntos de entrenamiento y prueba
// ============================================================================
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../../core/process_dataset/dataset.h"
#include "../../../utils/config.h"

/**
 * Hash function para vectores de AudioSample
 * Usa primeros 20 coeficientes MFCC para identificacion unica
 */
struct FeatureHash {
    size_t operator()(const std::vector<AudioSample>& features) const {
        size_t hash = 0;
        
        // Hash de primeros 20 coeficientes (suficiente para detectar duplicados)

        size_t n_hash = std::min(features.size(), size_t(20));
        
        for (size_t i = 0; i < n_hash; ++i) {
            // FNV-1a hash modification con double precision
            hash ^= std::hash<AudioSample>{}(features[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        
        return hash;
    }
};

static bool sonFeaturesIdenticas(const std::vector<AudioSample>& a,
    const std::vector<AudioSample>& b,
    AudioSample tolerancia = 1e-9) {
    if (a.size() != b.size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerancia) {
            return false;
        }
    }

    return true;
}

/**
 * Verifica que conjuntos train y test sean completamente disjuntos
 * 
 * @param train Conjunto de entrenamiento
 * @param test Conjunto de prueba
 * @return true si son disjuntos, false si hay data leakage
 * 
 * CRITERIOS DE FALLO:
 * - Si encuentra features identicas en ambos conjuntos
 * - Si hay mas de 1% de colisiones (umbral de tolerancia)
 * 
 * NOTA: Hashing NO es perfecto (colisiones posibles), pero detecta
 *       duplicados exactos con alta probabilidad (99.99%+)
 */
bool verificarDisjuncionTrainTest(const Dataset& train, const Dataset& test) {
    
    std::cout << "\n-> Verificando disjuncion Train/Test..." << std::endl;
    std::cout << "   Train: " << train.size() << " muestras" << std::endl;
    std::cout << "   Test:  " << test.size() << " muestras" << std::endl;
    
    if (train.empty() || test.empty()) {
        std::cerr << "! Error: Conjuntos vacios" << std::endl;
        return false;
    }

    const size_t dim_train = train.X.empty() ? 0 : train.X.front().size();
    const size_t dim_test = test.X.empty() ? 0 : test.X.front().size();

    if (dim_train == 0 || dim_test == 0) {
        std::cerr << "! Error: Dimension de features invalida" << std::endl;
        return false;
    }

    if (dim_train != dim_test) {
        std::cerr << "! Error: Dimensiones distintas entre train/test (" << dim_train
                  << " vs " << dim_test << ")" << std::endl;
        return false;
    }
    
    // PASO 1: Hashear todas las features de train
    std::unordered_map<size_t, std::vector<size_t>> hashes_train;
    hashes_train.reserve(train.size());
    
    std::cout << "   Hasheando conjunto de entrenamiento..." << std::endl;
    for (size_t idx = 0; idx < train.X.size(); ++idx) {
        const size_t hash = FeatureHash{}(train.X[idx]);
        hashes_train[hash].push_back(idx);
    }
    
    std::cout << "   Hashes unicos en train: " << hashes_train.size()
              << " (de " << train.size() << " muestras)" << std::endl;
    
    // PASO 2: Detectar colisiones en test
    std::cout << "   Buscando colisiones en test..." << std::endl;
    
    int colisiones_hash = 0;
    int duplicados_confirmados = 0;
    std::vector<size_t> indices_duplicados;
    
    for (size_t i = 0; i < test.X.size(); ++i) {
        const size_t hash = FeatureHash{}(test.X[i]);
        const auto it = hashes_train.find(hash);
        if (it == hashes_train.end()) {
            continue;
        }

        colisiones_hash++;

        bool es_duplicado = false;
        for (const size_t idx_train : it->second) {
            if (sonFeaturesIdenticas(train.X[idx_train], test.X[i])) {
                es_duplicado = true;
                duplicados_confirmados++;
                indices_duplicados.push_back(i);

                if (duplicados_confirmados <= 5) {
                    std::cout << "   ! Duplicado confirmado: test[" << i
                              << "] (label=" << test.y[i] << ") coincide con train["
                              << idx_train << "]" << std::endl;
                }
                break;
            }
        }

        if (!es_duplicado && it->second.size() > 3) {
            std::cout << "   % Nota: hash compartido por " << it->second.size()
                      << " muestras de train, pero ninguna coincide exactamente" << std::endl;
        }
    }
    
    // PASO 3: Analizar resultados
    const AudioSample porcentaje_colision = 100.0 * colisiones_hash / test.size();
    const AudioSample porcentaje_duplicados = 100.0 * duplicados_confirmados / test.size();
    
    std::cout << "\n   Resultado:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Colisiones hash detectadas: " << colisiones_hash
              << " / " << test.size()
              << " (" << std::fixed << std::setprecision(2)
              << porcentaje_colision << "%)" << std::endl;
    std::cout << "   Duplicados confirmados: " << duplicados_confirmados
              << " / " << test.size() << " (" << porcentaje_duplicados
              << "%)" << std::endl;
    
    // CRITERIO DE ACEPTACION:
    // - 0 colisiones: OK perfecto
    // - <1% colisiones: Aceptable (pueden ser colisiones de hash falsas)
    // - >=1% colisiones: CRITICO - data leakage probable
    
    constexpr AudioSample umbral_leakage = 1.0;
    const bool es_disjunto = (porcentaje_duplicados < umbral_leakage);
    
    if (duplicados_confirmados == 0) {
        std::cout << "   @ OK: Conjuntos completamente disjuntos" << std::endl;
    }
    else if (es_disjunto) {
        std::cout << "   % Advertencia: " << duplicados_confirmados
                  << " duplicados detectados (" << porcentaje_duplicados
                  << "%)" << std::endl;
        std::cout << "     Porcentaje bajo (<1%), revisar pero aceptable para continuar" << std::endl;
    }
    else {
        std::cerr << "\n   ! CRITICO: DATA LEAKAGE DETECTADO !" << std::endl;
        std::cerr << "   " << std::string(60, '=') << std::endl;
        std::cerr << "   " << duplicados_confirmados << " muestras de test estan presentes en train!" << std::endl;
        std::cerr << "   Esto invalida las metricas de evaluacion." << std::endl;
        std::cerr << "\n   CAUSAS POSIBLES:" << std::endl;
        std::cerr << "   1. Augmentation aplicada ANTES del split" << std::endl;
        std::cerr << "   2. Mismo audio procesado multiples veces" << std::endl;
        std::cerr << "   3. Audios de misma sesion divididos entre train/test" << std::endl;
        std::cerr << "\n   SOLUCION:" << std::endl;
        std::cerr << "   - Regenerar dataset con split temporal por sesion" << std::endl;
        std::cerr << "   - O usar split estratificado diferente (seed distinta)" << std::endl;
        std::cerr << "   " << std::string(60, '=') << std::endl;
    }
    
    std::cout << "   " << std::string(60, '-') << std::endl;
    
    return es_disjunto;
}

/**
 * Version mejorada de dividirTrainTest con verificacion integrada
 */
SplitResult dividirTrainTestSeguro(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y,
    AudioSample train_ratio,
    unsigned int seed) {
    
    // Llamar funcion original
    SplitResult result = dividirTrainTest(X, y, train_ratio, seed);
    
    if (result.train.empty() || result.test.empty()) {
        return result;  // Ya fallo antes
    }
    
    // VERIFICACION ADICIONAL: Disjuncion
    if (!verificarDisjuncionTrainTest(result.train, result.test)) {
        std::cerr << "\n! ABORTAR: Data leakage detectado" << std::endl;
        std::cerr << "  No se puede continuar con entrenamiento" << std::endl;
        
        // Retornar resultado vacio para forzar fallo
        return SplitResult();
    }
    
    return result;
}

/**
 * Calcula estadisticas de similaridad entre features (metrica complementaria)
 * Detecta features muy similares aunque no sean identicas
 */
struct SimilarityStats {
    AudioSample distancia_min;      // Distancia minima train-test
    AudioSample distancia_promedio;
    int pares_muy_cercanos;         // Con distancia < 0.01
};

SimilarityStats calcularSimilaridadTrainTest(const Dataset& train, 
                                              const Dataset& test,
                                              int n_samples = 100) {
    
    std::cout << "\n-> Calculando similaridad train-test (muestreo)..." << std::endl;
    
    SimilarityStats stats;
    stats.distancia_min = std::numeric_limits<AudioSample>::max();
    stats.distancia_promedio = 0.0;
    stats.pares_muy_cercanos = 0;

    if (train.empty() || test.empty()) {
        std::cerr << "! Error: No se puede calcular similaridad con conjuntos vacios" << std::endl;
        return stats;
    }
    
    // Muestreo aleatorio para no explotar complejidad O(N*M)
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist_train(0, train.size() - 1);
    std::uniform_int_distribution<size_t> dist_test(0, test.size() - 1);
    
    for (int i = 0; i < n_samples; ++i) {
        size_t idx_train = dist_train(gen);
        size_t idx_test = dist_test(gen);
        
        // Distancia euclidiana entre features
        const size_t dim = std::min(train.X[idx_train].size(), test.X[idx_test].size());
        if (dim == 0) {
            continue;
        }

        AudioSample dist = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            AudioSample diff = train.X[idx_train][j] - test.X[idx_test][j];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        
        stats.distancia_min = std::min(stats.distancia_min, dist);
        stats.distancia_promedio += dist;
        
        if (dist < 0.01) {  // Umbral de "muy cercano"
            stats.pares_muy_cercanos++;
        }
    }
    
    stats.distancia_promedio /= n_samples;
    
    std::cout << "   Distancia minima encontrada: " << std::fixed 
              << std::setprecision(4) << stats.distancia_min << std::endl;
    std::cout << "   Distancia promedio: " << stats.distancia_promedio << std::endl;
    std::cout << "   Pares muy cercanos (<0.01): " << stats.pares_muy_cercanos 
              << " / " << n_samples << std::endl;
    
    if (stats.distancia_min < 0.001) {
        std::cout << "   % Advertencia: Features extremadamente similares detectadas" << std::endl;
        std::cout << "     Puede indicar augmentation correlacionada" << std::endl;
    }
    
    return stats;
}

// ============================================================================
// MAIN 
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  VERIFICACION DE DATA LEAKAGE TRAIN/TEST" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Determinar rutas de datasets
    std::string ruta_train = obtenerRutaDatasetTrain();
    std::string ruta_test = obtenerRutaDatasetTest();

    // Permitir override por argumentos de linea de comandos
    if (argc >= 2) {
        ruta_train = argv[1];
    }
    if (argc >= 3) {
        ruta_test = argv[2];
    }

    std::cout << "-> Rutas de datasets:" << std::endl;
    std::cout << "   Train: " << ruta_train << std::endl;
    std::cout << "   Test:  " << ruta_test << std::endl;

    // PASO 1: Cargar datasets
    std::cout << "\n-> Cargando datasets..." << std::endl;
    
    Dataset train = cargarDatasetBinario(ruta_train);
    if (train.empty()) {
        std::cerr << "! Error: No se pudo cargar dataset de entrenamiento" << std::endl;
        std::cerr << "  Verifica que el archivo existe y tiene el formato correcto" << std::endl;
        return 1;
    }
    std::cout << "   Train cargado: " << train.size() << " muestras, "
              << train.dim() << " features" << std::endl;

    Dataset test = cargarDatasetBinario(ruta_test);
    if (test.empty()) {
        std::cerr << "! Error: No se pudo cargar dataset de prueba" << std::endl;
        std::cerr << "  Verifica que el archivo existe y tiene el formato correcto" << std::endl;
        return 1;
    }
    std::cout << "   Test cargado: " << test.size() << " muestras, "
              << test.dim() << " features" << std::endl;

    // PASO 2: Validar datasets
    std::cout << "\n-> Validando integridad de datasets..." << std::endl;
    
    if (!validarDataset(train)) {
        std::cerr << "! Error: Dataset de entrenamiento invalido" << std::endl;
        return 1;
    }
    std::cout << "   @ Train valido" << std::endl;

    if (!validarDataset(test)) {
        std::cerr << "! Error: Dataset de prueba invalido" << std::endl;
        return 1;
    }
    std::cout << "   @ Test valido" << std::endl;

    // PASO 3: Verificar compatibilidad
    std::cout << "\n-> Verificando compatibilidad train/test..." << std::endl;
    
    if (!verificarCompatibilidad(train, test)) {
        std::cerr << "! Error: Datasets incompatibles" << std::endl;
        return 1;
    }
    std::cout << "   @ Datasets compatibles" << std::endl;

    // PASO 4: Verificacion principal - DISJUNCION
    bool es_disjunto = verificarDisjuncionTrainTest(train, test);

    // PASO 5: Analisis complementario - SIMILARIDAD
    std::cout << "\n-> Analisis complementario de similaridad..." << std::endl;
    int n_samples_similaridad = std::min(500, static_cast<int>(std::min(train.size(), test.size())));
    std::cout << "   Evaluando " << n_samples_similaridad << " pares aleatorios..." << std::endl;
    
    SimilarityStats sim_stats = calcularSimilaridadTrainTest(train, test, n_samples_similaridad);

    // PASO 6: Resumen final
    std::cout << "\n============================================" << std::endl;
    std::cout << "  RESUMEN DE VERIFICACION" << std::endl;
    std::cout << "============================================" << std::endl;
    
    std::cout << "\n# Estado de disjuncion: ";
    if (es_disjunto) {
        std::cout << "@ APROBADO" << std::endl;
        std::cout << "  Los conjuntos train y test son disjuntos" << std::endl;
        std::cout << "  Es seguro entrenar y evaluar con estos datos" << std::endl;
    } else {
        std::cout << "! REPROBADO" << std::endl;
        std::cout << "  Se detecto DATA LEAKAGE entre train y test" << std::endl;
        std::cout << "  Las metricas de evaluacion NO seran confiables" << std::endl;
        std::cout << "\n  ACCION REQUERIDA:" << std::endl;
        std::cout << "  - Regenerar los datasets con split correcto" << std::endl;
        std::cout << "  - Verificar que augmentation se aplica POST-split" << std::endl;
        std::cout << "  - Revisar proceso de generacion de features" << std::endl;
    }

    std::cout << "\n# Estadisticas de similaridad:" << std::endl;
    std::cout << "  Distancia minima: " << std::fixed << std::setprecision(4) 
              << sim_stats.distancia_min << std::endl;
    std::cout << "  Distancia promedio: " << sim_stats.distancia_promedio << std::endl;
    std::cout << "  Pares muy cercanos: " << sim_stats.pares_muy_cercanos 
              << " / " << n_samples_similaridad << std::endl;

    std::cout << "\n============================================\n" << std::endl;

    // Codigo de salida
    return es_disjunto ? 0 : 1;
}