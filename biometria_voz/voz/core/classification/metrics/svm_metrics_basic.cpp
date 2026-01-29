// svm_metrics_basic.cpp - METRICAS TRADICIONALES DE CLASIFICACION
// Metricas basicas: Accuracy, Precision, Recall, F1, Specificity, MCC
// Matrices de confusion y estadisticas multiclase
#include "svm_metrics.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>

// ============================================================================
// CONSTRUCTORES DE ESTRUCTURAS
// ============================================================================

MatrizConfusion::MatrizConfusion(int tp, int tn, int fp, int fn)
    : tp(tp), tn(tn), fp(fp), fn(fn) {

    int total = tp + tn + fp + fn;
    if (total == 0) {
        accuracy = precision = recall = specificity = f1_score = mcc = 0.0;
        fpr = fnr = 0.0;
        return;
    }

    // Accuracy
    accuracy = 100.0 * (tp + tn) / total;

    // Precision
    precision = (tp + fp > 0) ? 100.0 * tp / (tp + fp) : 0.0;

    // Recall (Sensitivity)
    recall = (tp + fn > 0) ? 100.0 * tp / (tp + fn) : 0.0;

    // Specificity
    specificity = (tn + fp > 0) ? 100.0 * tn / (tn + fp) : 0.0;

    // F1-Score
    f1_score = (precision + recall > 0)
        ? 2.0 * (precision * recall) / (precision + recall) : 0.0;

    // MCC (Matthews Correlation Coefficient)
    AudioSample numerator = static_cast<AudioSample>(tp * tn - fp * fn);
    AudioSample denominator = std::sqrt(static_cast<AudioSample>(tp + fp) * (tp + fn) *
        (tn + fp) * (tn + fn));
    mcc = (denominator > 0) ? numerator / denominator : 0.0;

    // False Positive Rate
    fpr = (tn + fp > 0) ? 100.0 * fp / (tn + fp) : 0.0;

    // False Negative Rate
    fnr = (tp + fn > 0) ? 100.0 * fn / (tp + fn) : 0.0;
}

MatrizConfusion::MatrizConfusion()
    : tp(0), tn(0), fp(0), fn(0),
    accuracy(0), precision(0), recall(0), specificity(0),
    f1_score(0), mcc(0), fpr(0), fnr(0) {
}

// ============================================================================
// CALCULO DE METRICAS BASICAS
// ============================================================================

Metricas calcularMetricas(const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    int clasePositiva) {

    MatrizConfusion mc = calcularMatrizConfusion(y_real, y_pred, clasePositiva);

    Metricas m;
    m.accuracy = mc.accuracy;
    m.precision = mc.precision;
    m.recall = mc.recall;
    m.f1_score = mc.f1_score;
    m.specificity = mc.specificity;

    return m;
}

MatrizConfusion calcularMatrizConfusion(const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    int clasePositiva) {

    if (y_real.size() != y_pred.size()) {
        std::cerr << "! Error: Vectores de tamanos diferentes en calcularMatrizConfusion"
            << std::endl;
        return MatrizConfusion();
    }

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < y_real.size(); ++i) {
        bool real_pos = (y_real[i] == clasePositiva);
        bool pred_pos = (y_pred[i] == clasePositiva);

        if (real_pos && pred_pos) tp++;
        else if (!real_pos && !pred_pos) tn++;
        else if (!real_pos && pred_pos) fp++;
        else fn++;
    }

    return MatrizConfusion(tp, tn, fp, fn);
}

std::map<int, std::map<int, int>> calcularMatrizConfusionMulticlase(
    const std::vector<int>& y_real,
    const std::vector<int>& y_pred) {

    if (y_real.size() != y_pred.size()) {
        std::cerr << "! Error: Vectores de tamanos diferentes" << std::endl;
        return std::map<int, std::map<int, int>>();
    }

    std::map<int, std::map<int, int>> matriz;

    for (size_t i = 0; i < y_real.size(); ++i) {
        matriz[y_real[i]][y_pred[i]]++;
    }

    return matriz;
}

EstadisticasMulticlase calcularEstadisticasMulticlase(
    const std::vector<int>& y_real,
    const std::vector<int>& y_pred,
    const std::vector<int>& clases) {

    EstadisticasMulticlase stats;
    stats.total_muestras = static_cast<int>(y_real.size());
    stats.total_correctos = 0;

    // Calcular metricas por clase (macro-averaging)
    AudioSample sum_accuracy = 0.0;
    AudioSample sum_precision = 0.0;
    AudioSample sum_recall = 0.0;
    AudioSample sum_f1 = 0.0;
    AudioSample sum_specificity = 0.0;
    AudioSample sum_mcc = 0.0;

    for (int clase : clases) {
        MatrizConfusion mc = calcularMatrizConfusion(y_real, y_pred, clase);

        sum_accuracy += mc.accuracy;
        sum_precision += mc.precision;
        sum_recall += mc.recall;
        sum_f1 += mc.f1_score;
        sum_specificity += mc.specificity;
        sum_mcc += mc.mcc;
    }

    int num_clases = static_cast<int>(clases.size());
    stats.accuracy_promedio = sum_accuracy / num_clases;
    stats.precision_promedio = sum_precision / num_clases;
    stats.recall_promedio = sum_recall / num_clases;
    stats.f1_promedio = sum_f1 / num_clases;
    stats.specificity_promedio = sum_specificity / num_clases;
    stats.mcc_promedio = sum_mcc / num_clases;

    // Contar correctos e incorrectos
    for (size_t i = 0; i < y_real.size(); ++i) {
        if (y_real[i] == y_pred[i]) {
            stats.total_correctos++;
        }
        else {
            stats.errores_por_clase[y_real[i]]++;
        }
    }

    stats.total_incorrectos = stats.total_muestras - stats.total_correctos;

    return stats;
}

// ============================================================================
// VISUALIZACION DE METRICAS BASICAS
// ============================================================================

void mostrarMetricas(const Metricas& m, const std::string& nombre) {
    std::cout << "\n-> Metricas para: " << nombre << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   Accuracy:    " << std::fixed << std::setprecision(2)
        << m.accuracy << "%" << std::endl;
    std::cout << "   Precision:   " << std::setprecision(2)
        << m.precision << "%" << std::endl;
    std::cout << "   Recall:      " << std::setprecision(2)
        << m.recall << "%" << std::endl;
    std::cout << "   Specificity: " << std::setprecision(2)
        << m.specificity << "%" << std::endl;
    std::cout << "   F1-Score:    " << std::setprecision(2)
        << m.f1_score << "%" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;
}

void mostrarMatrizConfusion(const MatrizConfusion& mc, int claseId) {
    std::cout << "\n-> Matriz de Confusion - Clase " << claseId << std::endl;
    std::cout << "   " << std::string(40, '-') << std::endl;
    std::cout << "                 Pred: Pos  |  Pred: Neg" << std::endl;
    std::cout << "   Real: Pos      " << std::setw(5) << mc.tp
        << "      |    " << std::setw(5) << mc.fn << std::endl;
    std::cout << "   Real: Neg      " << std::setw(5) << mc.fp
        << "      |    " << std::setw(5) << mc.tn << std::endl;
    std::cout << "   " << std::string(40, '-') << std::endl;
    std::cout << "   Accuracy: " << std::fixed << std::setprecision(2)
        << mc.accuracy << "%" << std::endl;
    std::cout << "   " << std::string(40, '-') << std::endl;
}

void mostrarMatrizConfusionExtendida(const MatrizConfusion& mc, int claseId) {
    std::cout << "\n-> Matriz de Confusion Extendida - Clase " << claseId << std::endl;
    std::cout << "   " << std::string(50, '=') << std::endl;

    // Tabla de confusion
    std::cout << "   Matriz:" << std::endl;
    std::cout << "                 Pred: Pos  |  Pred: Neg" << std::endl;
    std::cout << "   Real: Pos      " << std::setw(5) << mc.tp
        << "      |    " << std::setw(5) << mc.fn << std::endl;
    std::cout << "   Real: Neg      " << std::setw(5) << mc.fp
        << "      |    " << std::setw(5) << mc.tn << std::endl;

    std::cout << "\n   Metricas basicas:" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   Accuracy:    " << std::fixed << std::setprecision(2)
        << mc.accuracy << "%" << std::endl;
    std::cout << "   Precision:   " << mc.precision << "%" << std::endl;
    std::cout << "   Recall:      " << mc.recall << "%" << std::endl;
    std::cout << "   Specificity: " << mc.specificity << "%" << std::endl;
    std::cout << "   F1-Score:    " << mc.f1_score << "%" << std::endl;

    std::cout << "\n   Metricas avanzadas:" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   MCC (Matthews): " << std::setprecision(4)
        << mc.mcc << std::endl;
    std::cout << "   FPR (False Pos Rate):  " << std::setprecision(2)
        << mc.fpr << "%" << std::endl;
    std::cout << "   FNR (False Neg Rate):  " << mc.fnr << "%" << std::endl;

    std::cout << "   " << std::string(50, '=') << std::endl;
}

void mostrarMatrizConfusionMulticlase(
    const std::map<int, std::map<int, int>>& matriz,
    const std::vector<int>& clases) {

    std::cout << "\n-> Matriz de Confusion Multiclase" << std::endl;
    std::cout << "   " << std::string(60, '=') << std::endl;

    // Encabezado
    std::cout << "   Real\\Pred  ";
    for (int clase : clases) {
        std::cout << std::setw(6) << clase;
    }
    std::cout << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;

    // Filas
    for (int clase_real : clases) {
        std::cout << "   " << std::setw(8) << clase_real << "  ";

        for (int clase_pred : clases) {
            auto it_real = matriz.find(clase_real);
            int count = 0;

            if (it_real != matriz.end()) {
                auto it_pred = it_real->second.find(clase_pred);
                if (it_pred != it_real->second.end()) {
                    count = it_pred->second;
                }
            }

            std::cout << std::setw(6) << count;
        }
        std::cout << std::endl;
    }

    std::cout << "   " << std::string(60, '=') << std::endl;
}

void mostrarEstadisticasMulticlase(const EstadisticasMulticlase& stats) {
    std::cout << "\n-> Estadisticas Multiclase (Macro-Average)" << std::endl;
    std::cout << "   " << std::string(60, '=') << std::endl;

    std::cout << "\n   Metricas promedio:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Accuracy:    " << std::fixed << std::setprecision(2)
        << stats.accuracy_promedio << "%" << std::endl;
    std::cout << "   Precision:   " << stats.precision_promedio << "%" << std::endl;
    std::cout << "   Recall:      " << stats.recall_promedio << "%" << std::endl;
    std::cout << "   Specificity: " << stats.specificity_promedio << "%" << std::endl;
    std::cout << "   F1-Score:    " << stats.f1_promedio << "%" << std::endl;
    std::cout << "   MCC:         " << std::setprecision(4)
        << stats.mcc_promedio << std::endl;

    std::cout << "\n   Resumen global:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Total muestras:      " << stats.total_muestras << std::endl;
    std::cout << "   Correctos:           " << stats.total_correctos
        << " (" << std::fixed << std::setprecision(2)
        << (100.0 * stats.total_correctos / stats.total_muestras) << "%)"
        << std::endl;
    std::cout << "   Incorrectos:         " << stats.total_incorrectos
        << " (" << std::fixed << std::setprecision(2)
        << (100.0 * stats.total_incorrectos / stats.total_muestras) << "%)"
        << std::endl;

    if (!stats.errores_por_clase.empty()) {
        std::cout << "\n   Errores por clase:" << std::endl;
        std::cout << "   " << std::string(60, '-') << std::endl;

        for (const auto& [clase, errores] : stats.errores_por_clase) {
            std::cout << "   Clase " << std::setw(5) << clase
                << ": " << std::setw(3) << errores << " errores" << std::endl;
        }
    }

    std::cout << "   " << std::string(60, '=') << std::endl;
}

// ============================================================================
// EVALUACION COMPLETA (CONSOLIDADO DESDE entrenar_modelo.cpp)
// ============================================================================

// Forward declaration - necesaria porque usa predecirHablante de svm.h
int predecirHablante(const std::vector<AudioSample>& x, const ModeloSVM& modelo);

void evaluarModeloCompleto(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y_real,
    const ModeloSVM& modelo,
    const std::string& titulo) {

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "EVALUACION COMPLETA - " << titulo << std::endl;

    // IMPORTANTE: X ya debe venir con expansion polinomial aplicada si esta activada
    // (se aplica en entrenar_modelo.cpp antes de llamar a esta funcion)
    
    // Generar predicciones
    std::vector<int> y_pred(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        y_pred[i] = predecirHablante(X[i], modelo);
    }

    // Calcular matriz de confusion multiclase
    auto matriz = calcularMatrizConfusionMulticlase(y_real, y_pred);

    // Mostrar solo errores (confusiones)
    std::cout << "\n-> Confusiones detectadas (Real -> Predicho):" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;

    int totalErrores = 0;
    int totalMuestras = static_cast<int>(y_real.size());

    for (const auto& [clase_real, predicciones] : matriz) {
        for (const auto& [clase_pred, count] : predicciones) {
            if (clase_real != clase_pred && count > 0) {
                std::cout << "   Hablante " << std::setw(5) << clase_real
                    << " -> " << std::setw(5) << clase_pred
                    << " : " << std::setw(2) << count << " error(es)" << std::endl;
                totalErrores += count;
            }
        }
    }

    if (totalErrores == 0) {
        std::cout << "   @ No hay confusiones! Clasificacion perfecta." << std::endl;
    }

    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   @ Total errores: " << totalErrores << "/" << totalMuestras
        << " (" << std::fixed << std::setprecision(2)
        << (100.0 * totalErrores / totalMuestras) << "%)" << std::endl;
    std::cout << "   @ Accuracy: " << std::fixed << std::setprecision(2)
        << (100.0 * (totalMuestras - totalErrores) / totalMuestras) << "%"
        << std::endl;

    // Mostrar hablantes mas problematicos
    std::cout << "\n-> Hablantes con errores de clasificacion:" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;

    std::map<int, int> erroresPorClase;
    std::map<int, int> muestrasPorClase;

    for (size_t i = 0; i < y_real.size(); ++i) {
        muestrasPorClase[y_real[i]]++;
        if (y_real[i] != y_pred[i]) {
            erroresPorClase[y_real[i]]++;
        }
    }

    bool hayErrores = false;
    for (const auto& [clase, errores] : erroresPorClase) {
        if (errores > 0) {
            hayErrores = true;
            int total = muestrasPorClase[clase];
            AudioSample errorRate = 100.0 * errores / total;
            std::cout << "   Hablante " << std::setw(5) << clase
                << ": " << std::setw(2) << errores << "/" << total
                << " errores (" << std::fixed << std::setprecision(1)
                << errorRate << "%)" << std::endl;
        }
    }

    if (!hayErrores) {
        std::cout << "   @ Ningun hablante tiene errores!" << std::endl;
    }

    // Calcular estadisticas multiclase
    EstadisticasMulticlase stats = calcularEstadisticasMulticlase(
        y_real, y_pred, modelo.clases);

    std::cout << "\n-> Metricas:" << std::endl;
    std::cout << "   " << std::string(50, '-') << std::endl;
    std::cout << "   Precision promedio:   " << std::fixed << std::setprecision(2)
        << stats.precision_promedio << "%" << std::endl;
    std::cout << "   Recall promedio:      " << stats.recall_promedio << "%" << std::endl;
    std::cout << "   F1-Score promedio:    " << stats.f1_promedio << "%" << std::endl;
    std::cout << "   Specificity promedio: " << stats.specificity_promedio << "%"
        << std::endl;
}