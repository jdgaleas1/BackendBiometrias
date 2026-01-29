#ifndef SVM_H
#define SVM_H

#include <vector>
#include <string>
#include <map>
#include <random>
#include <numeric>
#include "config.h"

// Forward declaration de estructuras biometricas (definidas en metrics/svm_metrics.h)
struct MetricasBiometricas;
struct PuntoROC;
struct CurvaROC; 

// ============================================================================
// ESTRUCTURAS PUBLICAS - VERSION 3.0 REFACTORIZADA
// Migracion: float -> AudioSample (double) para precision biometrica
// ============================================================================

/**
 * Estructura que representa un clasificador binario individual
 * Usa kernel lineal: decision(x) = w·x + b
 */
struct ClasificadorBinario {
    std::vector<AudioSample> pesos;  // Vector w de pesos
    AudioSample bias;                 // Sesgo b

    // Calibracion de probabilidades (Platt Scaling)
    AudioSample plattA;  // Parametro A para sigmoid
    AudioSample plattB;  // Parametro B para sigmoid

    // Threshold adaptativo (optimizado durante entrenamiento)
    AudioSample thresholdOptimo;  // Umbral que maximiza F1

    ClasificadorBinario() : bias(0.0), plattA(1.0), plattB(0.0), thresholdOptimo(0.0) {}
};

/**
 * Estructura que representa un modelo SVM One-vs-All entrenado
 * Contiene un clasificador binario lineal por cada clase
 * Kernel lineal: decision(x) = w·x + b
 */
struct ModeloSVM {
    std::vector<int> clases;

    // Parametros del clasificador lineal
    std::vector<std::vector<AudioSample>> pesosPorClase;  // [clase][feature]
    std::vector<AudioSample> biasPorClase;                 // [clase]

    // Calibracion de probabilidades (Platt Scaling)
    std::vector<AudioSample> plattAPorClase;    // [clase]
    std::vector<AudioSample> plattBPorClase;    // [clase]

    // Thresholds adaptativos (optimizados durante entrenamiento)
    std::vector<AudioSample> thresholdsPorClase;  // [clase]

    int dimensionCaracteristicas;

    ModeloSVM() : dimensionCaracteristicas(0) {}
};


/**
 * Metricas de evaluacion para clasificacion binaria
 */
struct Metricas {
    AudioSample accuracy;
    AudioSample precision;
    AudioSample recall;
    AudioSample f1_score;
    AudioSample specificity;
};

/**
 * Matriz de confusion para clasificacion binaria
 * Incluye metricas derivadas y tasas de error
 */
struct MatrizConfusion {
    int tp;
    int tn;
    int fp;
    int fn;
    AudioSample accuracy;
    AudioSample precision;
    AudioSample recall;
    AudioSample specificity;
    AudioSample f1_score;
    AudioSample mcc;
    AudioSample fpr;
    AudioSample fnr;

    MatrizConfusion(int tp, int tn, int fp, int fn);
    MatrizConfusion();
};

/**
 * Estadisticas agregadas para clasificacion multiclase
 * Promedia metricas sobre todas las clases (macro-averaging)
 */
struct EstadisticasMulticlase {
    AudioSample accuracy_promedio;
    AudioSample precision_promedio;
    AudioSample recall_promedio;
    AudioSample f1_promedio;
    AudioSample specificity_promedio;
    AudioSample mcc_promedio;
    int total_muestras;
    int total_correctos;
    int total_incorrectos;
    std::map<int, int> errores_por_clase;
};

// ============================================================================
// API CORE - PREDICCION Y SCORING
// ============================================================================

/**
 * Predice la clase de una muestra usando el modelo SVM entrenado
 *
 * @param x Vector de caracteristicas de entrada
 * @param modelo Modelo SVM entrenado
 * @return ID de la clase predicha, -1 si hay error
 *
 * Algoritmo:
 * - Calcula score para cada clasificador One-vs-All
 * - Retorna la clase con mayor score (decision function)
 */
int predecirHablante(const std::vector<AudioSample>& x, const ModeloSVM& modelo);

/**
 * Obtiene los scores de todas las clases para una muestra
 *
 * @param x Vector de caracteristicas de entrada
 * @param modelo Modelo SVM entrenado
 * @return Vector con scores por clase (mismo orden que modelo.clases)
 *
 * Util para analisis de confianza y debugging:
 * - Score alto positivo = fuerte prediccion para esa clase
 * - Score bajo negativo = fuerte rechazo de esa clase
 * - Scores similares = baja confianza en la prediccion
 */
std::vector<AudioSample> obtenerScores(const std::vector<AudioSample>& x,
    const ModeloSVM& modelo);

// ============================================================================
// API TRAIN - ENTRENAMIENTO
// ============================================================================

/**
 * Entrena un modelo SVM One-vs-All para clasificacion multiclase
 * Usa configuracion centralizada desde CONFIG_SVM (config.h)
 *
 * @param X Matriz de caracteristicas [muestras][dimensiones]
 * @param y Vector de etiquetas de clase
 * @return ModeloSVM entrenado con un clasificador One-vs-All por clase
 *
 * Estrategia de entrenamiento:
 * - One-vs-All: Entrena un clasificador binario por cada clase
 * - Ponderacion adaptativa: Ajusta pesos segun desbalance de clases
 * - SGD con momentum: Optimizacion con gradiente descendente estocastico
 * - Early stopping: Detiene entrenamiento al alcanzar specificity objetivo
 * - Regularizacion L2: Previene overfitting
 */
ModeloSVM entrenarSVMOVA(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y);

// ============================================================================
// API I/O - MODELOS SVM
// ============================================================================

/**
 * Guarda un modelo SVM entrenado en formato binario
 *
 * @param ruta Ruta del archivo de salida
 * @param modelo Modelo SVM a guardar
 * @return true si se guardo correctamente
 *
 * Formato binario compacto que incluye:
 * - Numero de clases
 * - Dimension de caracteristicas
 * - Pesos y bias por cada clase
 */
bool guardarModeloSVM(const std::string& ruta, const ModeloSVM& modelo);

/**
 * Carga un modelo SVM desde archivo binario
 *
 * @param ruta Ruta del archivo del modelo
 * @return ModeloSVM cargado (vacio si hay error)
 *
 * El archivo debe haber sido creado con guardarModeloSVM()
 */
ModeloSVM cargarModeloSVM(const std::string& ruta);

// ============================================================================
// API I/O - MODELOS MODULARES (INCREMENTAL)
// ============================================================================

/**
 * Guarda un clasificador binario individual en formato binario
 * Kernel lineal: guarda pesos w, bias b y parametros de calibracion
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param clase ID del usuario/clase
 * @param clasificador Clasificador binario lineal
 * @return true si se guardo exitosamente
 */
bool guardarClasificadorBinario(
    const std::string& ruta_base,
    int clase,
    const ClasificadorBinario& clasificador
);

/**
 * Carga un clasificador binario individual desde archivo
 * Kernel lineal: carga pesos w, bias b y parametros de calibracion
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param clase ID del usuario/clase a cargar
 * @param clasificador Clasificador binario cargado (salida)
 * @return true si se cargo exitosamente
 */
bool cargarClasificadorBinario(
    const std::string& ruta_base,
    int clase,
    ClasificadorBinario& clasificador
);

/**
 * Guarda metadata del modelo en formato JSON
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param num_clases Numero total de clases
 * @param dimension Dimension de caracteristicas
 * @param clases Vector con IDs de todas las clases
 * @return true si se guardo exitosamente
 */
bool guardarMetadata(
    const std::string& ruta_base,
    int num_clases,
    int dimension,
    const std::vector<int>& clases
);

/**
 * Carga metadata del modelo desde archivo JSON
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param num_clases Numero de clases (salida)
 * @param dimension Dimension de caracteristicas (salida)
 * @param clases Vector con IDs de clases (salida)
 * @return true si se cargo exitosamente
 */
bool cargarMetadata(
    const std::string& ruta_base,
    int& num_clases,
    int& dimension,
    std::vector<int>& clases
);

/**
 * Carga modelo completo desde directorio con archivos individuales
 * Reconstruye ModeloSVM a partir de metadata + clasificadores individuales
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @return ModeloSVM cargado (vacio si error)
 */
ModeloSVM cargarModeloModular(const std::string& ruta_base);

/**
 * Guarda modelo completo en archivos individuales
 * Crea: metadata.json + speaker_X.bin por cada clase
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param modelo Modelo SVM a guardar
 * @return true si se guardo exitosamente
 */
bool guardarModeloModular(const std::string& ruta_base, const ModeloSVM& modelo);



// ============================================================================
// NOTA: I/O DE DATASETS MOVIDO A process_dataset/dataset.h
// ============================================================================
// Las funciones cargarDatasetBinario() y guardarDatasetBinario() estan
// centralizadas en el modulo process_dataset para evitar duplicacion.
// Incluir dataset.h si necesitas usar estas funciones.
// ============================================================================

// ============================================================================
// API UTILS - UTILIDADES
// ============================================================================

/**
 * Expande features con terminos cuadraticos para aproximar kernel polinomial grado 2
 * Transforma: [x1, x2, ..., xn] -> [x1, x2, ..., xn, x1², x2², ..., xn²]
 * 
 * Esto permite al SVM lineal aprender fronteras de decision cuadraticas no lineales.
 * IMPORTANTE: Debe aplicarse CONSISTENTEMENTE en entrenamiento e inferencia.
 * 
 * @param X Matriz de caracteristicas original (sera modificada in-place)
 */
void expandirFeaturesPolinomial(std::vector<std::vector<AudioSample>>& X);

/**
 * Realiza diagnostico completo de un dataset
 * Analiza distribucion de clases, desbalance y validez de datos
 *
 * @param X Matriz de caracteristicas
 * @param y Vector de etiquetas
 *
 * Verifica:
 * - Numero de muestras y dimensiones
 * - Distribucion de clases y porcentajes
 * - Ratio de desbalance (max/min muestras)
 * - Presencia de NaN o Inf en los datos
 */
void diagnosticarDataset(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y);

/**
 * Calcula el producto punto entre dos vectores (centralizado)
 * Usado por svm_core.cpp y svm_train.cpp
 *
 * @param a Primer vector
 * @param b Segundo vector
 * @return Producto punto a·b
 *
 * Complejidad: O(n) donde n = dimension
 * Optimizado: std::inner_product usa SIMD cuando esta disponible
 */
inline AudioSample dotProduct(const std::vector<AudioSample>& a,
    const std::vector<AudioSample>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// ============================================================================
// INCLUIR API DE METRICAS
// ============================================================================
// Las funciones de metricas estan en metrics/svm_metrics.h
#include "metrics/svm_metrics.h"

#endif // SVM_H