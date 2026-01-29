#ifndef SVM_METRICAS_H
#define SVM_METRICAS_H

#include <vector>
#include <string>

struct ResultadosMetricas {
    double accuracy = 0.0;
    double precision_macro = 0.0;
    double recall_macro = 0.0;
    double f1_macro = 0.0;
    double precision_micro = 0.0;
    double recall_micro = 0.0;
    double f1_micro = 0.0;
    double precision_weighted = 0.0;
    double recall_weighted = 0.0;
    double f1_weighted = 0.0;
    double balanced_accuracy = 0.0;
    double mcc = 0.0;

    std::vector<std::vector<int>> matriz_confusion;
    std::vector<int>    soporte_por_clase;
    std::vector<double> precision_por_clase;
    std::vector<double> recall_por_clase;
    std::vector<double> f1_por_clase;
};

ResultadosMetricas calcularMetricasAvanzadas(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    int num_clases);

void exportarMetricasParaGraficos(const ResultadosMetricas& metricas,
    const std::string& basePath = "out/");

#endif
