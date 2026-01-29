#include "metricas/svm_metricas.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

ResultadosMetricas calcularMetricasAvanzadas(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    int K) {
    ResultadosMetricas res;

    res.matriz_confusion.assign(K, std::vector<int>(K, 0));
    res.soporte_por_clase.assign(K, 0);
    res.precision_por_clase.assign(K, 0.0);
    res.recall_por_clase.assign(K, 0.0);
    res.f1_por_clase.assign(K, 0.0);

    for (size_t i = 0; i < y_true.size(); ++i) {
        const int yi = y_true[i];
        const int yp = y_pred[i];
        if (yi >= 0 && yi < K && yp >= 0 && yp < K) {
            res.matriz_confusion[yi][yp] += 1;
            res.soporte_por_clase[yi] += 1;
        }
    }

    std::vector<int> sumFila(K, 0), sumCol(K, 0);
    int N = 0, tr = 0;
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) sumFila[i] += res.matriz_confusion[i][j];
        N += sumFila[i];
        tr += res.matriz_confusion[i][i];
    }
    for (int j = 0; j < K; ++j)
        for (int i = 0; i < K; ++i) sumCol[j] += res.matriz_confusion[i][j];

    res.accuracy = (N > 0) ? 100.0 * double(tr) / double(N) : 0.0;

    double precision_sum = 0.0, recall_sum = 0.0, f1_sum = 0.0, bacc_sum = 0.0;
    double precision_weighted_sum = 0.0, recall_weighted_sum = 0.0, f1_weighted_sum = 0.0;
    int soporte_total = std::accumulate(res.soporte_por_clase.begin(), res.soporte_por_clase.end(), 0);

    for (int i = 0; i < K; ++i) {
        const int TP = res.matriz_confusion[i][i];
        const int FN = sumFila[i] - TP;
        const int FP = sumCol[i] - TP;
        const int TN = N - TP - FP - FN;

        const double precision = (TP + FP) ? double(TP) / double(TP + FP) : 0.0;
        const double recall = (TP + FN) ? double(TP) / double(TP + FN) : 0.0;
        const double f1 = (precision + recall) ? 2.0 * precision * recall / (precision + recall) : 0.0;
        const double specificity = (TN + FP) ? double(TN) / double(TN + FP) : 0.0; // TNR
        const double bacc = 0.5 * (recall + specificity);

        res.precision_por_clase[i] = 100.0 * precision;
        res.recall_por_clase[i] = 100.0 * recall;
        res.f1_por_clase[i] = 100.0 * f1;

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
        bacc_sum += recall;

        const double w = (soporte_total > 0) ? double(res.soporte_por_clase[i]) / soporte_total : 0.0;
        precision_weighted_sum += w * precision;
        recall_weighted_sum += w * recall;
        f1_weighted_sum += w * f1;
    }

    res.precision_macro = (K > 0) ? 100.0 * precision_sum / K : 0.0;
    res.recall_macro = (K > 0) ? 100.0 * recall_sum / K : 0.0;
    res.f1_macro = (K > 0) ? 100.0 * f1_sum / K : 0.0;
    res.balanced_accuracy = (K > 0) ? 100.0 * bacc_sum / K : 0.0;

    res.precision_weighted = 100.0 * precision_weighted_sum;
    res.recall_weighted = 100.0 * recall_weighted_sum;
    res.f1_weighted = 100.0 * f1_weighted_sum;

    long long TPg = 0, FPg = 0, FNg = 0;
    for (int i = 0; i < K; ++i) {
        TPg += res.matriz_confusion[i][i];
        FPg += sumCol[i] - res.matriz_confusion[i][i];
        FNg += sumFila[i] - res.matriz_confusion[i][i];
    }
    const double prec_micro = (TPg + FPg) ? double(TPg) / double(TPg + FPg) : 0.0;
    const double rec_micro = (TPg + FNg) ? double(TPg) / double(TPg + FNg) : 0.0;
    const double f1_micro = (prec_micro + rec_micro) ? 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro) : 0.0;

    res.precision_micro = 100.0 * prec_micro;
    res.recall_micro = 100.0 * rec_micro;
    res.f1_micro = 100.0 * f1_micro;

    long long sumRowSq = 0, sumColSq = 0;
    for (int i = 0; i < K; ++i) {
        sumRowSq += 1LL * sumFila[i] * sumFila[i];
        sumColSq += 1LL * sumCol[i] * sumCol[i];
    }
    long long numer = 1LL * tr * N;
    for (int i = 0; i < K; ++i) numer -= 1LL * sumFila[i] * sumCol[i];

    long long denomL = 1LL * N * N - sumColSq;
    long long denomR = 1LL * N * N - sumRowSq;

    double mcc = 0.0;
    if (denomL > 0 && denomR > 0) {
        mcc = double(numer) / std::sqrt(double(denomL) * double(denomR));
    }
    res.mcc = 100.0 * mcc;

    return res;
}

void exportarMetricasParaGraficos(const ResultadosMetricas& metricas,
    const std::string& basePath) {
    fs::create_directories(basePath);

    {
        std::ofstream out(basePath + "/metricas_por_clase.csv");
        out << "Clase;Precision;Recall;F1;Soporte\n";
        out << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < metricas.precision_por_clase.size(); ++i) {
            out << i << ";" << metricas.precision_por_clase[i] << ";"
                << metricas.recall_por_clase[i] << ";"
                << metricas.f1_por_clase[i] << ";"
                << metricas.soporte_por_clase[i] << "\n";
        }
    }
    {
        std::ofstream conf(basePath + "/confusion_matrix.csv");
        conf << ";";
        for (size_t j = 0; j < metricas.matriz_confusion.size(); ++j)
            conf << "Pred_" << j << (j + 1 == metricas.matriz_confusion.size() ? "\n" : ";");

        for (size_t i = 0; i < metricas.matriz_confusion.size(); ++i) {
            conf << "Real_" << i << ";";
            for (size_t j = 0; j < metricas.matriz_confusion[i].size(); ++j) {
                conf << metricas.matriz_confusion[i][j];
                if (j + 1 != metricas.matriz_confusion[i].size()) conf << ";";
            }
            conf << "\n";
        }
    }
}
