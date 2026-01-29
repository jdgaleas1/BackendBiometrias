#ifndef SVM_METRICS_H
#define SVM_METRICS_H

#include "../svm.h"
#include <vector>
#include <string>
#include <map>

// NOTA: Las estructuras Metricas, MatrizConfusion y EstadisticasMulticlase
// estan definidas en ../svm.h para evitar dependencias circulares

// ============================================================================
// ESTRUCTURAS BIOMETRICAS
// ============================================================================

/**
 * Metricas biometricas para un clasificador binario
 * Usadas en sistemas de verificacion de identidad
 */
struct MetricasBiometricas {
    // Tasas de error biometrico
    AudioSample FAR;  // False Acceptance Rate (%)
    AudioSample FRR;  // False Rejection Rate (%)
    AudioSample EER;  // Equal Error Rate (%)
    
    // Threshold optimo
    AudioSample threshold_eer;  // Umbral donde FAR = FRR
    
    // AUC (Area Under Curve)
    AudioSample AUC;  // Area bajo la curva ROC [0.0, 1.0]
    
    // Matriz de confusion en threshold actual
    int tp;
    int tn;
    int fp;
    int fn;
    
    MetricasBiometricas() 
        : FAR(0.0), FRR(0.0), EER(0.0), threshold_eer(0.0), 
          AUC(0.0), tp(0), tn(0), fp(0), fn(0) {}
};

/**
 * Punto en la curva ROC
 */
struct PuntoROC {
    AudioSample threshold;    // Umbral de decision
    AudioSample tpr;          // True Positive Rate (Recall/Sensitivity)
    AudioSample fpr;          // False Positive Rate (1 - Specificity)
    AudioSample far;          // False Acceptance Rate
    AudioSample frr;          // False Rejection Rate
    
    PuntoROC() : threshold(0.0), tpr(0.0), fpr(0.0), far(0.0), frr(0.0) {}
};

/**
 * Curva ROC completa con multiples thresholds
 */
struct CurvaROC {
    std::vector<PuntoROC> puntos;
    AudioSample AUC;
    AudioSample EER;
    AudioSample threshold_eer;
    
    CurvaROC() : AUC(0.0), EER(0.0), threshold_eer(0.0) {}
};

// ============================================================================
// API BASICA - METRICAS TRADICIONALES
// ============================================================================

/**
 * Calcula metricas de evaluacion para una clase especifica (One-vs-All)
 *
 * @param y_real Vector con etiquetas reales
 * @param y_pred Vector con etiquetas predichas
 * @param clasePositiva Clase a evaluar (el resto se considera negativa)
 * @return Estructura Metricas con accuracy, precision, recall, F1, specificity
 */
Metricas calcularMetricas(const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    int clasePositiva);

/**
 * Calcula matriz de confusion para clasificacion binaria (One-vs-All)
 *
 * @param y_real Vector con etiquetas reales
 * @param y_pred Vector con etiquetas predichas
 * @param clasePositiva Clase positiva (el resto es negativo)
 * @return MatrizConfusion con TP, TN, FP, FN y metricas derivadas
 */
MatrizConfusion calcularMatrizConfusion(const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    int clasePositiva);

/**
 * Calcula matriz de confusion completa para clasificacion multiclase
 *
 * @param y_real Vector con etiquetas reales
 * @param y_pred Vector con etiquetas predichas
 * @return Mapa de mapas: matriz[clase_real][clase_pred] = conteo
 */
std::map<int, std::map<int, int>> calcularMatrizConfusionMulticlase(
    const std::vector<int>& y_real,
    const std::vector<int>& y_pred
);

/**
 * Calcula estadisticas agregadas para clasificacion multiclase
 * Usa macro-averaging (promedio de metricas por clase)
 *
 * @param y_real Vector con etiquetas reales
 * @param y_pred Vector con etiquetas predichas
 * @param clases Vector con todas las clases
 * @return EstadisticasMulticlase con promedios y conteos globales
 */
EstadisticasMulticlase calcularEstadisticasMulticlase(
    const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    const std::vector<int>& clases
);

/**
 * Imprime metricas en formato legible
 *
 * @param m Estructura Metricas a mostrar
 * @param nombre Nombre descriptivo (ej: "Clase 5", "Train", "Test")
 */
void mostrarMetricas(const Metricas& m, const std::string& nombre);

/**
 * Imprime matriz de confusion en formato de tabla
 *
 * @param mc Matriz de confusion a mostrar
 * @param claseId ID de la clase positiva
 */
void mostrarMatrizConfusion(const MatrizConfusion& mc, int claseId);

/**
 * Imprime matriz de confusion con todas las metricas derivadas
 *
 * @param mc Matriz de confusion a mostrar
 * @param claseId ID de la clase positiva
 *
 * Incluye: Accuracy, Precision, Recall, Specificity, F1, MCC, FPR, FNR
 */
void mostrarMatrizConfusionExtendida(const MatrizConfusion& mc, int claseId);

/**
 * Imprime matriz de confusion multiclase en formato tabular
 *
 * @param matriz Matriz de confusion multiclase
 * @param clases Vector con IDs de todas las clases
 */
void mostrarMatrizConfusionMulticlase(
    const std::map<int, std::map<int, int>>& matriz,
    const std::vector<int>& clases
);

/**
 * Imprime estadisticas multiclase en formato detallado
 *
 * @param stats Estadisticas a mostrar
 */
void mostrarEstadisticasMulticlase(const EstadisticasMulticlase& stats);

// ============================================================================
// API BIOMETRICA - METRICAS ROBUSTAS
// ============================================================================

/**
 * Calcula metricas biometricas para un clasificador binario
 * Incluye FAR, FRR, EER y AUC calculados desde la curva ROC
 *
 * @param scores Vector de scores de decision (uno por muestra)
 * @param y_binario Vector binario: +1 para positivos genuinos, -1 para impostores
 * @param num_thresholds Numero de umbrales para la curva ROC (default: 200)
 * @return Estructura MetricasBiometricas con FAR, FRR, EER, AUC
 *
 * Uso tipico en verificacion biometrica:
 * - Positivos (+1): Usuario genuino intentando autenticarse
 * - Negativos (-1): Impostor intentando hacerse pasar por el usuario
 */
MetricasBiometricas calcularMetricasBiometricas(
    const std::vector<AudioSample>& scores,
    const std::vector<int>& y_binario,
    int num_thresholds = 200
);

/**
 * Calcula curva ROC completa con multiples thresholds
 *
 * @param scores Vector de scores de decision
 * @param y_binario Vector binario: +1 para genuinos, -1 para impostores
 * @param num_thresholds Numero de puntos en la curva (default: 200)
 * @return CurvaROC con puntos (threshold, TPR, FPR, FAR, FRR), AUC y EER
 *
 * La curva ROC permite visualizar el trade-off entre:
 * - TPR (True Positive Rate): % de genuinos correctamente aceptados
 * - FPR (False Positive Rate): % de impostores incorrectamente aceptados
 */
CurvaROC calcularCurvaROC(
    const std::vector<AudioSample>& scores,
    const std::vector<int>& y_binario,
    int num_thresholds = 200
);

/**
 * Exporta curva ROC a archivo CSV para analisis/visualizacion
 *
 * @param roc Curva ROC calculada
 * @param ruta_archivo Ruta del archivo CSV de salida
 * @param clase_id ID de la clase (opcional, para nombrar archivo)
 * @return true si se exporto exitosamente
 *
 * Formato CSV:
 * threshold,TPR,FPR,FAR,FRR
 * -5.0,1.0,1.0,100.0,0.0
 * ...
 */
bool exportarROC_CSV(
    const CurvaROC& roc,
    const std::string& ruta_archivo,
    int clase_id = -1
);

/**
 * Imprime metricas biometricas en formato legible
 *
 * @param mb Metricas biometricas a mostrar
 * @param nombre Nombre descriptivo (ej: "Clase 5", "Usuario 1001")
 *
 * Muestra:
 * - FAR (False Acceptance Rate): % de impostores aceptados
 * - FRR (False Rejection Rate): % de genuinos rechazados
 * - EER (Equal Error Rate): Punto optimo donde FAR = FRR
 * - AUC (Area Under Curve): Calidad general del clasificador
 */
void mostrarMetricasBiometricas(
    const MetricasBiometricas& mb,
    const std::string& nombre
);

/**
 * Encuentra el threshold optimo para un objetivo especifico
 *
 * @param roc Curva ROC calculada
 * @param objetivo_far FAR objetivo (default: 1.0% = alta seguridad)
 * @return Threshold que minimiza FRR manteniendo FAR <= objetivo
 *
 * Uso: En sistemas biometricos, se fija un FAR maximo tolerable
 * (ej: 1% de impostores aceptados) y se busca el threshold que
 * minimiza FRR (rechazos de usuarios genuinos) bajo esa restriccion.
 */
AudioSample encontrarThresholdOptimo(
    const CurvaROC& roc,
    AudioSample objetivo_far = 1.0
);

// ============================================================================
// EVALUACION COMPLETA (DESDE entrenar_modelo.cpp)
// ============================================================================

/**
 * Evalua un modelo de forma completa mostrando matriz de confusion detallada
 * Consolidacion de la logica de entrenar_modelo.cpp
 *
 * @param X Matriz de caracteristicas
 * @param y_real Vector con etiquetas reales
 * @param modelo Modelo SVM entrenado
 * @param titulo Titulo descriptivo (ej: "ENTRENAMIENTO", "PRUEBA")
 *
 * Muestra:
 * - Matriz de confusion completa
 * - Errores por clase
 * - Hablantes problematicos
 * - Estadisticas globales
 */
void evaluarModeloCompleto(
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_real,
    const ModeloSVM& modelo,
    const std::string& titulo
);

#endif // SVM_METRICS_H