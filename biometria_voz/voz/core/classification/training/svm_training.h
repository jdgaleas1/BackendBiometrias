// svm_training.h - API PUBLICA DEL MODULO DE ENTRENAMIENTO
// Version 3.0 - Modularizado y refactorizado
#ifndef SVM_TRAINING_H
#define SVM_TRAINING_H

#include "../svm.h"
#include <vector>

// ============================================================================
// ESTRUCTURAS INTERNAS DEL ENTRENAMIENTO
// ============================================================================

/**
 * Resultado del entrenamiento de un clasificador binario
 * Contiene los parametros entrenados y metricas finales
 */
struct ResultadoEntrenamiento {
    std::vector<AudioSample> pesos;       // Vector w de pesos
    AudioSample bias;                      // Sesgo b
    
    // Metricas tradicionales finales
    AudioSample f1_final;
    AudioSample recall_final;
    AudioSample specificity_final;
    AudioSample precision_final;
    
    // Matriz de confusion final
    int tp;
    int tn;
    int fp;
    int fn;
    
    // Metricas biometricas robustas
    bool metricas_biometricas_validas;
    AudioSample FAR;              // False Acceptance Rate
    AudioSample FRR;              // False Rejection Rate
    AudioSample EER;              // Equal Error Rate
    AudioSample AUC;              // Area Under Curve
    AudioSample threshold_eer;    // Threshold optimo en EER
    
    // Estado del entrenamiento
    bool entrenamiento_exitoso;
    int epocas_realizadas;
    
    // Datos para exportar ROC (opcional)
    std::vector<AudioSample> scores_finales;
    std::vector<int> y_binario_final;
    
    ResultadoEntrenamiento() 
        : bias(0.0), f1_final(0.0), recall_final(0.0), 
          specificity_final(0.0), precision_final(0.0),
          tp(0), tn(0), fp(0), fn(0),
          metricas_biometricas_validas(false),
          FAR(0.0), FRR(0.0), EER(0.0), AUC(0.0), threshold_eer(0.0),
          entrenamiento_exitoso(false), epocas_realizadas(0) {}
};

// ============================================================================
// API PUBLICA - ENTRENAMIENTO
// ============================================================================

/**
 * Entrena un modelo SVM One-vs-All para clasificacion multiclase
 * 
 * @param X Matriz de caracteristicas [muestras][dimensiones]
 * @param y Vector de etiquetas de clase
 * @return ModeloSVM entrenado con un clasificador One-vs-All por clase
 * 
 * COMPATIBLE CON VERSION ANTERIOR - NO ROMPE API EXTERNA
 */
ModeloSVM entrenarSVMOVA(const std::vector<std::vector<AudioSample>>& X,
                         const std::vector<int>& y);

/**
 * Entrena SOLO un clasificador binario para una clase nueva (incremental)
 * 
 * @param ruta_modelo_base Directorio base del modelo (ej: "model/")
 * @param X Matriz de caracteristicas (TODO el dataset)
 * @param y Vector de etiquetas (TODO el dataset)
 * @param nueva_clase ID del nuevo usuario a entrenar
 * @return true si se entreno y guardo exitosamente
 * 
 * COMPATIBLE CON VERSION ANTERIOR - NO ROMPE API EXTERNA
 */
bool entrenarClaseIncremental(
    const std::string& ruta_modelo_base,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y,
    int nueva_clase
);

// ============================================================================
// API INTERNA - FUNCION CORE (NO EXPONER AL EXTERIOR)
// ============================================================================

/**
 * FUNCION CORE: Entrena un clasificador binario SVM
 * Esta funcion contiene TODA la logica de entrenamiento
 * Usada internamente por entrenarSVMOVA() y entrenarClaseIncremental()
 * 
 * @param X Matriz de caracteristicas
 * @param y_binario Vector binario: +1 para clase positiva, -1 para negativa
 * @param cfg Configuracion de entrenamiento SVM
 * @param seed Semilla para reproducibilidad
 * @return ResultadoEntrenamiento con pesos, bias y metricas
 * 
 * NOTA: Esta funcion NO debe ser llamada directamente desde fuera del modulo
 */
ResultadoEntrenamiento entrenarClasificadorBinario(
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_binario,
    const ConfigSVM& cfg,
    int seed
);

#endif // SVM_TRAINING_H