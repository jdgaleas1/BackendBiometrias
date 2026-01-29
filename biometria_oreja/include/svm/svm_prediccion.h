#ifndef SVM_PREDICCION_H
#define SVM_PREDICCION_H

#include "svm_entrenamiento.h"
#include <vector>
#include <string>

struct PrediccionOVA {
    int clase;
    double score;
    int clase2;
    double score2;
    std::vector<double> scores;
};

PrediccionOVA predecirConScores(const std::vector<double>& x, const ModeloSVM& modelo);
int predecirPersona(const std::vector<double>& x, const ModeloSVM& modelo);
int predecirPersonaConUmbral(const std::vector<double>& x, const ModeloSVM& modelo, double umbralScoreMinimo);
int predecirPersonaConMargen(const std::vector<double>& x, const ModeloSVM& modelo, double umbral_margen);
int predictOVAScore(const ModeloSVM &modelo, const std::vector<double> &x, double &bestScore, double &secondScore, int &bestClass);
bool enTopK(const PrediccionOVA& p, int y_true, int K, const ModeloSVM& modelo);

// DECLARACI�N (para que main pueda linkearla)
void evaluarModeloSimple(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const ModeloSVM& modelo,
    const std::string& nombre);

#endif
