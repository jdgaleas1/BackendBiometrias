// svm_train_binary.cpp - FUNCION CORE DE ENTRENAMIENTO BINARIO
// Esta funcion contiene TODA la logica de entrenamiento de un clasificador binario
// Es reutilizada por entrenarSVMOVA() y entrenarClaseIncremental()

#include "svm_training.h"
#include "../../../utils/config.h"
#include "../../../utils/additional.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <climits>
#include <cfloat>

// Declaraciones de funciones de utilidades
AudioSample calcularPesoClasePositiva(AudioSample ratio, const ConfigSVM& cfg);
bool detectarColapso(AudioSample recall, AudioSample specificity, const ConfigSVM& cfg);
std::vector<AudioSample> inicializarPesosXavier(int dimension, std::mt19937& gen);
void validarYAjustarModelo(
    const std::vector<AudioSample>& mejor_w,
    AudioSample& mejor_b,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_binario,
    bool modelo_guardado,
    const ConfigSVM& cfg
);

// ============================================================================
// ESTRUCTURA INTERNA PARA ESTADO DE OPTIMIZACION
// ============================================================================

struct EstadoOptimizador {
    // Variables de optimizacion
    std::vector<AudioSample> v_w;  // Velocidad/momentum para pesos
    std::vector<AudioSample> m_w;  // Momento de segundo orden (Adam)
    AudioSample v_b;               // Velocidad/momentum para bias
    AudioSample m_b;               // Momento de segundo orden para bias (Adam)
    int adam_t;                    // Contador de pasos para Adam
    
    // Learning rate
    AudioSample tasa_actual;
    AudioSample tasa_max;
    AudioSample tasa_min;
    
    // Control de mejora
    AudioSample mejor_f1;
    AudioSample mejor_loss;
    AudioSample mejor_recall;
    AudioSample mejor_specificity;
    int sin_mejora;
    bool modelo_guardado;
    
    // Control de convergencia
    AudioSample f1_anterior;
    int epocas_estancadas;
    
    // Control de colapso
    bool modelo_colapsado;
    int epocas_desde_colapso;
    int epocas_desde_reset;
    
    EstadoOptimizador(int dimension, AudioSample tasa_aprendizaje)
        : v_w(dimension, 0.0), m_w(dimension, 0.0),
          v_b(0.0), m_b(0.0), adam_t(0),
          tasa_max(tasa_aprendizaje), tasa_min(tasa_aprendizaje * 0.01),
          tasa_actual(tasa_aprendizaje),
          mejor_f1(0.0), mejor_loss(DBL_MAX),
          mejor_recall(0.0), mejor_specificity(0.0),
          sin_mejora(0), modelo_guardado(false),
          f1_anterior(0.0), epocas_estancadas(0),
          modelo_colapsado(false), epocas_desde_colapso(0), 
          epocas_desde_reset(10000) {}
};

// ============================================================================
// FUNCION AUXILIAR: APLICAR ACTUALIZACION DE GRADIENTE
// ============================================================================

void aplicarActualizacionGradiente(
    std::vector<AudioSample>& w,
    AudioSample& b,
    const std::vector<AudioSample>& grad_w,
    AudioSample grad_b,
    EstadoOptimizador& estado,
    int batch_size,
    AudioSample lambda,
    const ConfigSVM& cfg
) {
    if (cfg.usarAdamOptimizer) {
        // ADAM OPTIMIZER
        estado.adam_t++;
        
        // Actualizar pesos
        for (size_t j = 0; j < w.size(); ++j) {
            AudioSample g = grad_w[j] / batch_size + lambda * w[j];
            estado.v_w[j] = cfg.beta1Adam * estado.v_w[j] + (1.0 - cfg.beta1Adam) * g;
            estado.m_w[j] = cfg.beta2Adam * estado.m_w[j] + (1.0 - cfg.beta2Adam) * g * g;
            
            AudioSample v_hat = estado.v_w[j] / (1.0 - std::pow(cfg.beta1Adam, estado.adam_t));
            AudioSample m_hat = estado.m_w[j] / (1.0 - std::pow(cfg.beta2Adam, estado.adam_t));
            
            w[j] -= estado.tasa_actual * v_hat / (std::sqrt(m_hat) + cfg.epsilonAdam);
        }
        
        // Actualizar bias
        AudioSample g_b = grad_b / batch_size;
        estado.v_b = cfg.beta1Adam * estado.v_b + (1.0 - cfg.beta1Adam) * g_b;
        estado.m_b = cfg.beta2Adam * estado.m_b + (1.0 - cfg.beta2Adam) * g_b * g_b;
        
        AudioSample v_b_hat = estado.v_b / (1.0 - std::pow(cfg.beta1Adam, estado.adam_t));
        AudioSample m_b_hat = estado.m_b / (1.0 - std::pow(cfg.beta2Adam, estado.adam_t));
        
        b -= estado.tasa_actual * v_b_hat / (std::sqrt(m_b_hat) + cfg.epsilonAdam);
        
    } else {
        // SGD + MOMENTUM
        for (size_t j = 0; j < w.size(); ++j) {
            AudioSample g = grad_w[j] / batch_size + lambda * w[j];
            estado.v_w[j] = cfg.momentum * estado.v_w[j] + g;
            w[j] -= estado.tasa_actual * estado.v_w[j];
        }
        
        AudioSample g_b = grad_b / batch_size;
        estado.v_b = cfg.momentum * estado.v_b + g_b;
        b -= estado.tasa_actual * estado.v_b;
    }
    
    // Limitar bias para evitar valores extremos
    AudioSample bias_limit = cfg.usarAdamOptimizer ? -0.5 : -1.0;
    if (estado.epocas_desde_reset > 2000 && estado.sin_mejora > 500) {
        b = std::max(b, bias_limit);
    }
    estado.epocas_desde_reset++;
}

// ============================================================================
// FUNCION AUXILIAR: EVALUACION PERIODICA
// ============================================================================

bool evaluarYActualizarMejor(
    const std::vector<AudioSample>& w,
    AudioSample b,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_binario,
    AudioSample loss_total,
    std::vector<AudioSample>& mejor_w,
    AudioSample& mejor_b,
    EstadoOptimizador& estado,
    AudioSample& peso_positivo,
    int epoca,
    const ConfigSVM& cfg
) {
    int m = static_cast<int>(X.size());
    
    // Calcular matriz de confusion
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < m; ++i) {
        AudioSample score = b + dotProduct(w, X[i]);
        bool pred_pos = (score >= 0.0);
        bool real_pos = (y_binario[i] == 1);
        
        if (real_pos && pred_pos) tp++;
        else if (!real_pos && !pred_pos) tn++;
        else if (!real_pos && pred_pos) fp++;
        else fn++;
    }
    
    // Calcular metricas
    AudioSample recall = (tp + fn > 0) ? 100.0 * tp / (tp + fn) : 0.0;
    AudioSample specificity = (tn + fp > 0) ? 100.0 * tn / (tn + fp) : 0.0;
    AudioSample precision = (tp + fp > 0) ? 100.0 * tp / (tp + fp) : 0.0;
    AudioSample f1 = (precision + recall > 0)
        ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
    
    // DETECCION DE COLAPSO SEVERO
    if (epoca >= 500 && detectarColapso(recall, specificity, cfg)) {
        if (!estado.modelo_colapsado) {
            std::cout << "   % WARNING: Colapso severo en epoca " << epoca
                      << " (Rec=" << std::fixed << std::setprecision(1) << recall
                      << "% Spe=" << specificity << "%)" << std::endl;
            peso_positivo *= 0.6;
            estado.tasa_actual *= 0.6;
            estado.modelo_colapsado = true;
            
            // Restaurar mejor modelo si el colapso es extremo
            if (recall >= 99.5 && specificity < 10.0 && estado.mejor_f1 > 0.0) {
                // Nota: el caller debera copiar mejor_w a w
                estado.epocas_desde_reset = 0;
            }
        }
        estado.epocas_desde_colapso++;
        
        if (estado.epocas_desde_colapso > 1000) {
            std::cout << "   ! Modelo no se recupera, abortando clase" << std::endl;
            return false; // Señal de abortar
        }
    } else {
        estado.modelo_colapsado = false;
        estado.epocas_desde_colapso = 0;
    }
    
    // CRITERIOS DE GUARDADO DEL MODELO
    bool mejoro = false;
    AudioSample loss_norm = loss_total / m;
    bool metricas_objetivo = (specificity >= cfg.specificityTarget &&
                              recall >= cfg.recallMinimo &&
                              precision >= cfg.precisionMinima &&
                              f1 >= cfg.f1Minimo);
    
    AudioSample score_actual = f1 * 0.8 + precision * 0.15 + recall * 0.05;
    AudioSample mejor_score = estado.mejor_f1 * 0.8 + 
                              (estado.mejor_recall * 0.15) +
                              (estado.mejor_specificity * 0.05);
    
    bool debe_guardar = false;
    
    // Prioridades de guardado (misma logica que el original)
    if (loss_norm < 0.06 && f1 >= 40.0 && !estado.modelo_colapsado && epoca >= 400) {
        debe_guardar = true;
        estado.modelo_guardado = true;
    } else if (metricas_objetivo && score_actual > mejor_score && !estado.modelo_colapsado) {
        debe_guardar = true;
        estado.modelo_guardado = true;
    } else if (f1 >= 50.0 && f1 > estado.mejor_f1 + 1.0 && !estado.modelo_colapsado && epoca >= 200) {
        debe_guardar = true;
        estado.modelo_guardado = true;
    } else if (loss_norm < 0.08 && f1 >= 45.0 && loss_norm < estado.mejor_loss && !estado.modelo_colapsado && epoca >= 600) {
        debe_guardar = true;
        if (f1 >= cfg.f1Minimo) estado.modelo_guardado = true;
    } else if (f1 > estado.mejor_f1 && epoca >= 1500 && !estado.modelo_colapsado) {
        debe_guardar = true;
        if (f1 >= cfg.f1Minimo) estado.modelo_guardado = true;
    } else if (!estado.modelo_guardado && f1 > estado.mejor_f1 && epoca >= 3000) {
        debe_guardar = true;
    }
    
    if (debe_guardar) {
        estado.mejor_f1 = f1;
        estado.mejor_loss = loss_norm;
        mejor_w = w;
        mejor_b = b;
        estado.mejor_recall = recall;
        estado.mejor_specificity = specificity;
        estado.sin_mejora = 0;
        estado.epocas_desde_reset = 10000;
        mejoro = true;
    } else {
        estado.sin_mejora++;
    }
    
    // Deteccion de estancamiento
    if (epoca > 0 && (f1 - estado.f1_anterior) < 0.5) {
        estado.epocas_estancadas++;
    } else {
        estado.epocas_estancadas = 0;
    }
    estado.f1_anterior = f1;
    
    // IMPRIMIR PROGRESO (DESACTIVADO - solo logs internos)
    // std::cout << "   Epoca " << std::setw(5) << epoca
    //           << " | Loss=" << std::fixed << std::setprecision(3) << loss_norm
    //           << " | Rec=" << std::setprecision(1) << recall << "%"
    //           << " Spe=" << specificity << "%"
    //           << " Prec=" << precision << "%"
    //           << " F1=" << f1 << "%"
    //           << " | b=" << std::setprecision(3) << b;
    // 
    // if (mejoro) std::cout << " *";
    // if (loss_norm < 0.06 && f1 >= 40.0) std::cout << " [LOSS BAJO]";
    // if (estado.modelo_colapsado) std::cout << " [COLAPSO]";
    // if (estado.epocas_estancadas >= 10) std::cout << " [Estancado:" << estado.epocas_estancadas << "]";
    // std::cout << std::endl;
    
    return true; // Continuar entrenamiento
}

// ============================================================================
// FUNCION CORE: ENTRENAMIENTO DE CLASIFICADOR BINARIO
// ============================================================================

ResultadoEntrenamiento entrenarClasificadorBinario(
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_binario,
    const ConfigSVM& cfg,
    int seed
) {
    ResultadoEntrenamiento resultado;
    
    int m = static_cast<int>(X.size());
    int n = static_cast<int>(X[0].size());
    
    // Contar positivas y negativas
    int positivas = 0, negativas = 0;
    for (int label : y_binario) {
        if (label == 1) positivas++;
        else negativas++;
    }
    
    AudioSample ratio = static_cast<AudioSample>(negativas) / positivas;
    
    // Calcular pesos de clase
    AudioSample peso_positivo = calcularPesoClasePositiva(ratio, cfg);
    AudioSample peso_negativo = 1.0;
    
    // Inicializar pesos usando Xavier
    std::mt19937 gen(seed);
    std::vector<AudioSample> w = inicializarPesosXavier(n, gen);
    AudioSample b = 0.0;
    
    // Mejores pesos encontrados
    std::vector<AudioSample> mejor_w(n, 0.0);
    AudioSample mejor_b = -5.0;
    
    // Estado del optimizador
    EstadoOptimizador estado(n, cfg.tasaAprendizaje);
    AudioSample decay_rate = 0.9996;
    
    // Paciencia adaptativa
    int paciencia = (positivas < cfg.muestrasMinoritarias)
        ? cfg.pacienciaMinoritaria : cfg.paciencia;
    
    // Batch size adaptativo
    int batch_size = std::max(4, std::min(cfg.batchSizeNormal,
        static_cast<int>(positivas * 0.5)));
    
    // Indices para shuffle
    std::vector<int> indices_todos(m);
    std::iota(indices_todos.begin(), indices_todos.end(), 0);
    
    // ========================================================================
    // BUCLE PRINCIPAL DE ENTRENAMIENTO
    // ========================================================================
    
    for (int epoca = 0; epoca < cfg.epocas; ++epoca) {
        // Shuffle indices
        std::shuffle(indices_todos.begin(), indices_todos.end(), gen);
        
        AudioSample loss_total = 0.0;
        
        // MINI-BATCH SGD
        for (size_t start = 0; start < indices_todos.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, indices_todos.size());
            std::vector<int> batch_indices(indices_todos.begin() + start,
                indices_todos.begin() + end);
            
            // Acumuladores de gradiente
            std::vector<AudioSample> grad_w(n, 0.0);
            AudioSample grad_b = 0.0;
            
            // Calcular gradientes para el batch
            for (int idx : batch_indices) {
                int y_i = y_binario[idx];
                AudioSample score = b + dotProduct(w, X[idx]);
                AudioSample margin = 1.0 - y_i * score;
                AudioSample w_i = (y_i == 1) ? peso_positivo : peso_negativo;
                
                if (margin > 0.0) {
                    for (int j = 0; j < n; ++j) {
                        grad_w[j] += (-w_i * y_i * X[idx][j]);
                    }
                    grad_b += (-w_i * y_i);
                    loss_total += w_i * margin;
                }
            }
            
            // Aplicar actualizacion
            AudioSample lambda = 1.0 / (cfg.C * m);
            aplicarActualizacionGradiente(
                w, b, grad_w, grad_b, estado,
                static_cast<int>(batch_indices.size()),
                lambda, cfg
            );
        }
        
        // EVALUACION PERIODICA
        if (epoca % 200 == 0 || epoca == cfg.epocas - 1) {
            bool continuar = evaluarYActualizarMejor(
                w, b, X, y_binario, loss_total,
                mejor_w, mejor_b, estado, peso_positivo,
                epoca, cfg
            );
            
            if (!continuar) {
                resultado.entrenamiento_exitoso = false;
                resultado.epocas_realizadas = epoca;
                return resultado;
            }
            
            // EARLY STOPPING - Estancamiento
            if (estado.epocas_estancadas >= 25 && epoca >= cfg.epocasMinimas) {
                std::cout << "   @ Early stopping: ESTANCADO por "
                          << estado.epocas_estancadas << " eval (F1=" 
                          << std::fixed << std::setprecision(1) << estado.mejor_f1 << "%)" << std::endl;
                break;
            }
            
            // EARLY STOPPING - Objetivo alcanzado
            bool metricas_objetivo = (estado.mejor_specificity >= cfg.specificityTarget &&
                                      estado.mejor_recall >= cfg.recallMinimo &&
                                      estado.mejor_f1 >= cfg.f1Minimo);
            
            if ((estado.modelo_guardado && metricas_objetivo && 
                 estado.mejor_recall < cfg.umbralRecallColapso && epoca >= cfg.epocasMinimas) ||
                (epoca >= cfg.epocasMinimas * 2 && estado.mejor_f1 >= cfg.f1Minimo * 0.9)) {
                std::cout << "   @ Objetivo alcanzado en epoca " << epoca << std::endl;
                break;
            }
            
            // EARLY STOPPING - Sin mejora
            if (estado.sin_mejora >= paciencia && epoca >= cfg.epocasMinimas) {
                std::cout << "   @ Early stopping: sin mejora por " << estado.sin_mejora
                          << " evaluaciones (~" << (estado.sin_mejora * 200) << " epocas)" << std::endl;
                break;
            }
            
            // REDUCE LR ON PLATEAU
            if (estado.sin_mejora > 0 && estado.sin_mejora % 5 == 0) {
                estado.tasa_actual *= 0.6;
                estado.tasa_actual = std::max(estado.tasa_actual, estado.tasa_min);
            }
        }
        
        // DECAY EXPONENCIAL CONTINUO
        if (epoca > 0) {
            estado.tasa_actual *= decay_rate;
            estado.tasa_actual = std::max(estado.tasa_actual, estado.tasa_min);
        }
        
        resultado.epocas_realizadas = epoca + 1;
    }
    
    // ========================================================================
    // VALIDACION Y CALCULOS FINALES
    // ========================================================================
    
    validarYAjustarModelo(mejor_w, mejor_b, X, y_binario, estado.modelo_guardado, cfg);
    
    // Calcular metricas finales y scores para metricas biometricas
    int tp_final = 0, tn_final = 0, fp_final = 0, fn_final = 0;
    std::vector<AudioSample> scores_finales(m);
    
    for (int i = 0; i < m; ++i) {
        AudioSample score = mejor_b + dotProduct(mejor_w, X[i]);
        scores_finales[i] = score;
        
        bool pred_pos = (score >= 0.0);
        bool real_pos = (y_binario[i] == 1);
        
        if (real_pos && pred_pos) tp_final++;
        else if (!real_pos && !pred_pos) tn_final++;
        else if (!real_pos && pred_pos) fp_final++;
        else fn_final++;
    }
    
    // Metricas tradicionales
    resultado.recall_final = (tp_final + fn_final > 0) 
        ? 100.0 * tp_final / (tp_final + fn_final) : 0.0;
    resultado.specificity_final = (tn_final + fp_final > 0) 
        ? 100.0 * tn_final / (tn_final + fp_final) : 0.0;
    resultado.precision_final = (tp_final + fp_final > 0) 
        ? 100.0 * tp_final / (tp_final + fp_final) : 0.0;
    resultado.f1_final = (resultado.precision_final + resultado.recall_final > 0)
        ? 2.0 * (resultado.precision_final * resultado.recall_final) / 
          (resultado.precision_final + resultado.recall_final) : 0.0;
    
    // Matriz de confusion
    resultado.tp = tp_final;
    resultado.tn = tn_final;
    resultado.fp = fp_final;
    resultado.fn = fn_final;
    
    // Calcular metricas biometricas si esta habilitado
    if (cfg.imprimirMetricasRobustas) {
        try {
            MetricasBiometricas mb = calcularMetricasBiometricas(
                scores_finales, y_binario, 200
            );
            
            resultado.FAR = mb.FAR;
            resultado.FRR = mb.FRR;
            resultado.EER = mb.EER;
            resultado.AUC = mb.AUC;
            resultado.threshold_eer = mb.threshold_eer;
            resultado.metricas_biometricas_validas = true;
        } catch (...) {
            resultado.metricas_biometricas_validas = false;
        }
    }
    
    // Guardar curva ROC completa si esta habilitado (para analisis externo)
    resultado.scores_finales = scores_finales;
    resultado.y_binario_final = y_binario;
    
    // Asignar resultado
    resultado.pesos = mejor_w;
    resultado.bias = mejor_b;
    resultado.entrenamiento_exitoso = true;
    
    return resultado;
}