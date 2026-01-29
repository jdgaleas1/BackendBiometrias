// svm_metrics_robustas.cpp - METRICAS BIOMETRICAS ROBUSTAS
// Implementacion de metricas especializadas para sistemas biometricos:
// - FAR (False Acceptance Rate): Tasa de impostores aceptados
// - FRR (False Rejection Rate): Tasa de genuinos rechazados
// - EER (Equal Error Rate): Punto donde FAR = FRR
// - ROC Curve: Curva ROC (TPR vs FPR)
// - AUC: Area bajo la curva ROC
#include "svm_metrics.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// CALCULO DE CURVA ROC Y METRICAS DERIVADAS
// ============================================================================

/**
 * Calcula curva ROC completa con multiples thresholds
 * 
 * La curva ROC (Receiver Operating Characteristic) es fundamental en biometria
 * para analizar el trade-off entre seguridad (FPR bajo) y usabilidad (TPR alto).
 * 
 * Para cada threshold de decision:
 * - TPR (True Positive Rate) = TP / (TP + FN) = Recall/Sensitivity
 * - FPR (False Positive Rate) = FP / (FP + TN) = 1 - Specificity
 * - FAR (False Acceptance Rate) = FP / (FP + TN) = FPR (mismo concepto)
 * - FRR (False Rejection Rate) = FN / (FN + TP) = 1 - TPR
 * 
 * @param scores Vector de scores de decision (output de modelo SVM)
 * @param y_binario Vector binario: +1 genuinos, -1 impostores
 * @param num_thresholds Numero de puntos en la curva
 * @return CurvaROC con puntos, AUC y EER
 */
CurvaROC calcularCurvaROC(
    const std::vector<AudioSample>& scores,
    const std::vector<int>& y_binario,
    int num_thresholds
) {
    CurvaROC roc;
    
    if (scores.size() != y_binario.size()) {
        std::cerr << "! Error: Tamanos diferentes en calcularCurvaROC" << std::endl;
        return roc;
    }
    
    if (scores.empty()) {
        std::cerr << "! Error: Vector de scores vacio" << std::endl;
        return roc;
    }
    
    // ========================================================================
    // PASO 1: ENCONTRAR RANGO DE SCORES
    // ========================================================================
    
    AudioSample score_min = *std::min_element(scores.begin(), scores.end());
    AudioSample score_max = *std::max_element(scores.begin(), scores.end());
    
    // Expandir ligeramente el rango para incluir extremos
    AudioSample margen = (score_max - score_min) * 0.05;
    score_min -= margen;
    score_max += margen;
    
    // Validar que haya rango valido
    if (score_max - score_min < 1e-10) {
        std::cerr << "! Error: Todos los scores son identicos" << std::endl;
        return roc;
    }
    
    // ========================================================================
    // PASO 2: CONTAR POSITIVOS Y NEGATIVOS REALES
    // ========================================================================
    
    int P = 0;  // Total de genuinos (positivos reales)
    int N = 0;  // Total de impostores (negativos reales)
    
    for (int label : y_binario) {
        if (label == 1) P++;
        else N++;
    }
    
    if (P == 0 || N == 0) {
        std::cerr << "! Error: Se necesitan muestras positivas Y negativas" << std::endl;
        std::cerr << "   P=" << P << ", N=" << N << std::endl;
        return roc;
    }
    
    // ========================================================================
    // PASO 3: CALCULAR PUNTOS DE LA CURVA ROC
    // ========================================================================
    
    AudioSample step = (score_max - score_min) / (num_thresholds - 1);
    roc.puntos.reserve(num_thresholds);
    
    for (int i = 0; i < num_thresholds; ++i) {
        AudioSample threshold = score_min + i * step;
        
        // Contar TP, FP, TN, FN para este threshold
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (size_t j = 0; j < scores.size(); ++j) {
            bool pred_pos = (scores[j] >= threshold);  // Predicho como genuino
            bool real_pos = (y_binario[j] == 1);       // Genuino real
            
            if (real_pos && pred_pos) tp++;
            else if (!real_pos && !pred_pos) tn++;
            else if (!real_pos && pred_pos) fp++;
            else fn++;
        }
        
        // Calcular metricas para este threshold
        PuntoROC punto;
        punto.threshold = threshold;
        
        // TPR (True Positive Rate) = Recall = Sensitivity
        punto.tpr = (P > 0) ? (100.0 * tp / P) : 0.0;
        
        // FPR (False Positive Rate) = 1 - Specificity
        punto.fpr = (N > 0) ? (100.0 * fp / N) : 0.0;
        
        // FAR (False Acceptance Rate) = FPR en contexto biometrico
        punto.far = punto.fpr;
        
        // FRR (False Rejection Rate) = 1 - TPR
        punto.frr = 100.0 - punto.tpr;
        
        roc.puntos.push_back(punto);
    }
    
    // ========================================================================
    // PASO 4: CALCULAR AUC (AREA BAJO LA CURVA) - METODO TRAPEZOIDAL
    // ========================================================================
    
    // Ordenar puntos por FPR creciente para integracion correcta
    std::sort(roc.puntos.begin(), roc.puntos.end(),
        [](const PuntoROC& a, const PuntoROC& b) {
            return a.fpr < b.fpr;
        });
    
    AudioSample auc = 0.0;
    for (size_t i = 1; i < roc.puntos.size(); ++i) {
        // Formula trapezoidal: Area = (base * (altura1 + altura2)) / 2
        AudioSample base = (roc.puntos[i].fpr - roc.puntos[i-1].fpr) / 100.0;  // Normalizar a [0,1]
        AudioSample altura1 = roc.puntos[i-1].tpr / 100.0;
        AudioSample altura2 = roc.puntos[i].tpr / 100.0;
        auc += base * (altura1 + altura2) / 2.0;
    }
    
    roc.AUC = auc;
    
    // ========================================================================
    // PASO 5: ENCONTRAR EER (EQUAL ERROR RATE)
    // ========================================================================
    
    // EER = punto donde FAR = FRR (o donde TPR = 1 - FPR)
    // Buscamos el threshold donde |FAR - FRR| es minimo
    
    AudioSample min_diff = std::numeric_limits<AudioSample>::max();
    size_t eer_idx = 0;
    
    for (size_t i = 0; i < roc.puntos.size(); ++i) {
        AudioSample diff = std::abs(roc.puntos[i].far - roc.puntos[i].frr);
        if (diff < min_diff) {
            min_diff = diff;
            eer_idx = i;
        }
    }
    
    // EER es el promedio de FAR y FRR en el punto de cruce
    roc.EER = (roc.puntos[eer_idx].far + roc.puntos[eer_idx].frr) / 2.0;
    roc.threshold_eer = roc.puntos[eer_idx].threshold;
    
    return roc;
}

// ============================================================================
// METRICAS BIOMETRICAS DESDE SCORES
// ============================================================================

/**
 * Calcula metricas biometricas directamente desde scores del modelo
 * 
 * Esta funcion es el wrapper principal para obtener todas las metricas
 * biometricas relevantes en una sola llamada.
 * 
 * @param scores Vector de scores (decision function del SVM)
 * @param y_binario Vector binario: +1 genuinos, -1 impostores
 * @param num_thresholds Resolucion de la curva ROC
 * @return MetricasBiometricas con FAR, FRR, EER, AUC y matriz de confusion
 */
MetricasBiometricas calcularMetricasBiometricas(
    const std::vector<AudioSample>& scores,
    const std::vector<int>& y_binario,
    int num_thresholds
) {
    MetricasBiometricas mb;
    
    // Calcular curva ROC completa
    CurvaROC roc = calcularCurvaROC(scores, y_binario, num_thresholds);
    
    if (roc.puntos.empty()) {
        std::cerr << "! Error: No se pudo calcular curva ROC" << std::endl;
        return mb;
    }
    
    // Extraer metricas principales
    mb.EER = roc.EER;
    mb.threshold_eer = roc.threshold_eer;
    mb.AUC = roc.AUC;
    
    // Encontrar FAR y FRR en el threshold EER
    for (const auto& punto : roc.puntos) {
        if (std::abs(punto.threshold - roc.threshold_eer) < 1e-6) {
            mb.FAR = punto.far;
            mb.FRR = punto.frr;
            break;
        }
    }
    
    // Calcular matriz de confusion en threshold EER
    for (size_t i = 0; i < scores.size(); ++i) {
        bool pred_pos = (scores[i] >= mb.threshold_eer);
        bool real_pos = (y_binario[i] == 1);
        
        if (real_pos && pred_pos) mb.tp++;
        else if (!real_pos && !pred_pos) mb.tn++;
        else if (!real_pos && pred_pos) mb.fp++;
        else mb.fn++;
    }
    
    return mb;
}

// ============================================================================
// THRESHOLD OPTIMO PARA OBJETIVO FAR
// ============================================================================

/**
 * Encuentra el threshold optimo para un objetivo de FAR especifico
 * 
 * En sistemas biometricos de alta seguridad, se suele fijar un FAR maximo
 * tolerable (ej: 1% de impostores aceptados) y se busca el threshold que
 * minimiza FRR (rechazos de usuarios genuinos) bajo esa restriccion.
 * 
 * @param roc Curva ROC calculada
 * @param objetivo_far FAR maximo tolerable (%)
 * @return Threshold que minimiza FRR con FAR <= objetivo_far
 */
AudioSample encontrarThresholdOptimo(
    const CurvaROC& roc,
    AudioSample objetivo_far
) {
    if (roc.puntos.empty()) {
        std::cerr << "! Error: Curva ROC vacia" << std::endl;
        return 0.0;
    }
    
    // Buscar el threshold que minimiza FRR manteniendo FAR <= objetivo
    AudioSample mejor_threshold = roc.puntos[0].threshold;
    AudioSample min_frr = 100.0;
    
    for (const auto& punto : roc.puntos) {
        // Solo considerar puntos que cumplan el objetivo de FAR
        if (punto.far <= objetivo_far) {
            if (punto.frr < min_frr) {
                min_frr = punto.frr;
                mejor_threshold = punto.threshold;
            }
        }
    }
    
    return mejor_threshold;
}

// ============================================================================
// EXPORTACION DE CURVA ROC A CSV
// ============================================================================

/**
 * Exporta curva ROC a archivo CSV para analisis externo
 * 
 * El archivo CSV se puede usar con Python/R/Excel para:
 * - Visualizar graficamente la curva ROC
 * - Calcular metricas adicionales
 * - Comparar diferentes modelos
 * 
 * @param roc Curva ROC a exportar
 * @param ruta_archivo Ruta completa del archivo CSV
 * @param clase_id ID de la clase (opcional, para nombrar el archivo)
 * @return true si se exporto exitosamente
 */
bool exportarROC_CSV(
    const CurvaROC& roc,
    const std::string& ruta_archivo,
    int clase_id
) {
    std::ofstream out(ruta_archivo);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta_archivo << std::endl;
        return false;
    }
    
    // Encabezado CSV
    out << "threshold,TPR,FPR,FAR,FRR" << std::endl;
    
    // Escribir puntos de la curva
    out << std::fixed << std::setprecision(6);
    for (const auto& punto : roc.puntos) {
        out << punto.threshold << ","
            << punto.tpr << ","
            << punto.fpr << ","
            << punto.far << ","
            << punto.frr << std::endl;
    }
    
    out.close();
    
    std::string nombre_clase = (clase_id >= 0) 
        ? "Clase " + std::to_string(clase_id) 
        : "modelo";
    
    std::cout << "   & Curva ROC exportada: " << ruta_archivo << std::endl;
    std::cout << "      " << nombre_clase 
              << " | Puntos: " << roc.puntos.size()
              << " | AUC: " << std::setprecision(4) << roc.AUC
              << " | EER: " << std::setprecision(2) << roc.EER << "%" << std::endl;
    
    return true;
}

// ============================================================================
// VISUALIZACION DE METRICAS BIOMETRICAS
// ============================================================================

/**
 * Imprime metricas biometricas en formato legible y profesional
 * 
 * Contexto biometrico:
 * - FAR (False Acceptance Rate): % de IMPOSTORES que son ACEPTADOS
 *   * Metrica de SEGURIDAD: FAR bajo = sistema seguro
 *   * En produccion tipicamente se busca FAR < 0.1% - 1%
 * 
 * - FRR (False Rejection Rate): % de GENUINOS que son RECHAZADOS
 *   * Metrica de USABILIDAD: FRR bajo = sistema comodo
 *   * En produccion se tolera FRR de 1% - 5%
 * 
 * - EER (Equal Error Rate): Punto donde FAR = FRR
 *   * Metrica de BALANCE: EER bajo = buen equilibrio
 *   * EER < 5% es considerado bueno en biometria de voz
 * 
 * - AUC (Area Under Curve): Calidad general del clasificador
 *   * AUC cercano a 1.0 = clasificador excelente
 *   * AUC cercano a 0.5 = clasificador aleatorio
 * 
 * @param mb Metricas biometricas calculadas
 * @param nombre Identificador descriptivo
 */
void mostrarMetricasBiometricas(
    const MetricasBiometricas& mb,
    const std::string& nombre
) {
    std::cout << "\n-> Metricas Biometricas - " << nombre << std::endl;
    std::cout << "   " << std::string(60, '=') << std::endl;
    
    // Seccion 1: Metricas principales
    std::cout << "\n   [METRICAS CLAVE]" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    
    std::cout << "   FAR (False Accept Rate):   " << std::fixed << std::setprecision(2)
              << mb.FAR << "%" << std::endl;
    std::cout << "      -> Impostores aceptados (MENOR = MAS SEGURO)" << std::endl;
    
    std::cout << "\n   FRR (False Reject Rate):   " << std::setprecision(2)
              << mb.FRR << "%" << std::endl;
    std::cout << "      -> Genuinos rechazados (MENOR = MAS USABLE)" << std::endl;
    
    std::cout << "\n   EER (Equal Error Rate):    " << std::setprecision(2)
              << mb.EER << "%" << std::endl;
    std::cout << "      -> Balance FAR/FRR (MENOR = MEJOR)" << std::endl;
    std::cout << "      -> Threshold EER: " << std::setprecision(3)
              << mb.threshold_eer << std::endl;
    
    std::cout << "\n   AUC (Area Under Curve):    " << std::setprecision(4)
              << mb.AUC << std::endl;
    std::cout << "      -> Calidad del clasificador (CERCANO A 1.0 = EXCELENTE)" << std::endl;
    
    // Seccion 2: Matriz de confusion en threshold EER
    std::cout << "\n   [MATRIZ DE CONFUSION @ Threshold EER]" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "                 Pred: Genuino  |  Pred: Impostor" << std::endl;
    std::cout << "   Real: Genuino     " << std::setw(5) << mb.tp
              << "        |      " << std::setw(5) << mb.fn << std::endl;
    std::cout << "   Real: Impostor    " << std::setw(5) << mb.fp
              << "        |      " << std::setw(5) << mb.tn << std::endl;
    
    // Seccion 3: Interpretacion y recomendaciones
    std::cout << "\n   [INTERPRETACION]" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    
    // Calidad del modelo segun EER
    if (mb.EER < 1.0) {
        std::cout << "   @ EXCELENTE: EER < 1% (produccion lista)" << std::endl;
    } else if (mb.EER < 5.0) {
        std::cout << "   @ BUENO: EER < 5% (aceptable para produccion)" << std::endl;
    } else if (mb.EER < 10.0) {
        std::cout << "   % REGULAR: EER < 10% (requiere mejora)" << std::endl;
    } else {
        std::cout << "   ! POBRE: EER >= 10% (no recomendado para produccion)" << std::endl;
    }
    
    // Calidad segun AUC
    if (mb.AUC > 0.95) {
        std::cout << "   @ AUC excelente: Clasificador muy confiable" << std::endl;
    } else if (mb.AUC > 0.85) {
        std::cout << "   @ AUC bueno: Clasificador confiable" << std::endl;
    } else if (mb.AUC > 0.70) {
        std::cout << "   % AUC moderado: Clasificador aceptable" << std::endl;
    } else {
        std::cout << "   ! AUC bajo: Clasificador poco confiable" << std::endl;
    }
    
    // Balance FAR/FRR
    AudioSample ratio_far_frr = (mb.FRR > 0.01) ? mb.FAR / mb.FRR : 0.0;
    if (ratio_far_frr > 2.0) {
        std::cout << "   % Sesgo hacia SEGURIDAD (FAR << FRR)" << std::endl;
        std::cout << "      -> Muchos genuinos rechazados, pocos impostores aceptados" << std::endl;
    } else if (ratio_far_frr < 0.5) {
        std::cout << "   % Sesgo hacia USABILIDAD (FRR << FAR)" << std::endl;
        std::cout << "      -> Pocos genuinos rechazados, muchos impostores aceptados" << std::endl;
    } else {
        std::cout << "   @ Balance equilibrado entre seguridad y usabilidad" << std::endl;
    }
    
    std::cout << "   " << std::string(60, '=') << std::endl;
}