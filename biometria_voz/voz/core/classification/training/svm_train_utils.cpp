// svm_train_utils.cpp - UTILIDADES COMPARTIDAS DEL ENTRENAMIENTO
#include "svm_training.h"
#include "../../../utils/config.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>

// ============================================================================
// CALCULO DE PESOS PARA CLASES DESBALANCEADAS
// ============================================================================

/**
 * Calcula el peso adaptativo para la clase positiva basado en el ratio de desbalance
 * Usa estrategia logaritmica o raiz cuadrada segun configuracion
 * 
 * @param ratio Ratio negativas/positivas (ej: 10.0 significa 10:1)
 * @param cfg Configuracion SVM
 * @return Peso ajustado para clase positiva (clampeado entre min y max)
 * 
 * Estrategias:
 * - Logaritmica: peso = log(ratio + 1) * factor (suave, conservador)
 * - Raiz cuadrada: peso = sqrt(ratio) * factor (mas agresivo)
 */
AudioSample calcularPesoClasePositiva(AudioSample ratio, const ConfigSVM& cfg) {
    AudioSample peso;
    
    if (cfg.usarPesoLogaritmico) {
        peso = std::log(ratio + 1.0) * cfg.factorPesoConservador;
    } else {
        peso = std::sqrt(ratio) * cfg.factorPesoConservador;
    }
    
    return std::clamp(peso, cfg.pesoMinimo, cfg.pesoMaximo);
}

// ============================================================================
// DETECCION DE COLAPSO DEL MODELO
// ============================================================================

/**
 * Detecta si el modelo esta en colapso (predice todo como positivo)
 * 
 * Colapso = Recall muy alto (>=98%) + Specificity muy baja (<30%)
 * Indica que el modelo clasifica TODO como positivo (no discrimina)
 * 
 * @param recall Porcentaje de recall (sensibilidad)
 * @param specificity Porcentaje de specificity
 * @param cfg Configuracion SVM
 * @return true si el modelo esta colapsado
 */
bool detectarColapso(AudioSample recall, AudioSample specificity, const ConfigSVM& cfg) {
    return (recall >= cfg.umbralRecallColapso && specificity < 30.0);
}

// ============================================================================
// INICIALIZACION DE PESOS
// ============================================================================

/**
 * Inicializa pesos usando estrategia Xavier/Glorot
 * Formula: w ~ N(0, sqrt(2 / (n_in + n_out)))
 * 
 * Para SVM: n_out = 1 (salida escalar), entonces scale = sqrt(2 / (n + 1))
 * 
 * @param dimension Numero de features
 * @param gen Generador de numeros aleatorios
 * @return Vector de pesos inicializados
 */
std::vector<AudioSample> inicializarPesosXavier(int dimension, std::mt19937& gen) {
    std::vector<AudioSample> pesos(dimension);
    
    AudioSample init_scale = std::sqrt(2.0 / (dimension + 1));
    std::normal_distribution<AudioSample> init_dist(0.0, init_scale);
    
    for (int j = 0; j < dimension; ++j) {
        pesos[j] = init_dist(gen);
    }
    
    return pesos;
}

// ============================================================================
// VALIDACION DEL MODELO FINAL
// ============================================================================

/**
 * Valida el modelo entrenado y ajusta bias si es necesario
 * 
 * Criterios de validacion:
 * - F1-Score minimo
 * - Tasa de falsos positivos maxima
 * 
 * Si el modelo es muy malo, ajusta el bias hacia valores mas conservadores
 * 
 * @param mejor_w Mejores pesos encontrados
 * @param mejor_b Mejor bias encontrado (modificado si es necesario)
 * @param X Dataset completo
 * @param y_binario Etiquetas binarias
 * @param modelo_guardado Si se guardo al menos un modelo durante el entrenamiento
 * @param cfg Configuracion SVM
 */
void validarYAjustarModelo(
    const std::vector<AudioSample>& mejor_w,
    AudioSample& mejor_b,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_binario,
    bool modelo_guardado,
    const ConfigSVM& cfg
) {
    int m = static_cast<int>(X.size());
    
    // Contar positivas y negativas
    int positivas = 0, negativas = 0;
    for (int label : y_binario) {
        if (label == 1) positivas++;
        else negativas++;
    }
    
    // Calcular metricas finales
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < m; ++i) {
        AudioSample score = mejor_b + dotProduct(mejor_w, X[i]);
        bool pred_pos = (score >= 0.0);
        bool real_pos = (y_binario[i] == 1);
        
        if (real_pos && pred_pos) tp++;
        else if (!real_pos && !pred_pos) tn++;
        else if (!real_pos && pred_pos) fp++;
        else fn++;
    }
    
    AudioSample recall = (tp + fn > 0) ? 100.0 * tp / (tp + fn) : 0.0;
    AudioSample precision = (tp + fp > 0) ? 100.0 * tp / (tp + fp) : 0.0;
    AudioSample f1 = (precision + recall > 0) 
        ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
    
    // Validacion: modelo muy malo?
    int max_fp_permitido = static_cast<int>(negativas * 0.15);
    bool modelo_muy_malo = (f1 < 15.0 || fp > max_fp_permitido);
    
    if (!modelo_guardado && modelo_muy_malo) {
        std::cout << "   ! Modelo insuficiente: F1=" << std::fixed << std::setprecision(1) 
                  << f1 << "%, FP=" << fp << "/" << negativas 
                  << " (max=" << max_fp_permitido << ")" << std::endl;
        std::cout << "   ! Ajustando bias hacia valor mas conservador..." << std::endl;
        mejor_b = mejor_b - 1.5;
    }
}

// ============================================================================
// PREPARACION DE DATOS BINARIOS
// ============================================================================

/**
 * Convierte etiquetas multiclase a binarias (One-vs-All)
 * 
 * @param y Vector de etiquetas multiclase
 * @param clase_positiva Clase a considerar como positiva
 * @param positivas Contador de muestras positivas (salida)
 * @param negativas Contador de muestras negativas (salida)
 * @return Vector binario: +1 para clase_positiva, -1 para el resto
 */
std::vector<int> prepararDatosBinarios(
    const std::vector<int>& y,
    int clase_positiva,
    int& positivas,
    int& negativas
) {
    std::vector<int> y_binario(y.size());
    positivas = 0;
    negativas = 0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] == clase_positiva) {
            y_binario[i] = 1;
            positivas++;
        } else {
            y_binario[i] = -1;
            negativas++;
        }
    }
    
    return y_binario;
}