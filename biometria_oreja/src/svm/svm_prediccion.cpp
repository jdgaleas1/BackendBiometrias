#include "svm/svm_prediccion.h"
#include <numeric>
#include <limits>
#include <iostream>
#include <map>
#include <iomanip>
#include <algorithm>
#include <numeric>

PrediccionOVA predecirConScores(const std::vector<double>& x, const ModeloSVM& modelo) {
    PrediccionOVA r;
    r.clase = -1;
    r.score = -std::numeric_limits<double>::infinity();
    r.clase2 = -1;
    r.score2 = -std::numeric_limits<double>::infinity();
    r.scores.assign(modelo.clases.size(), -std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        const auto& w = modelo.pesosPorClase[i];

        // 🔥 guard obligatorio
        if (w.size() != x.size()) {
            continue; // clase inválida por mismatch
        }

        const double s = std::inner_product(x.begin(), x.end(), w.begin(), 0.0) + modelo.biasPorClase[i];
        r.scores[i] = s;

        if (s > r.score) {
            r.clase2 = r.clase;  r.score2 = r.score;
            r.clase = modelo.clases[i]; r.score = s;
        } else if (s > r.score2) {
            r.clase2 = modelo.clases[i]; r.score2 = s;
        }
    }
    return r;
}

int predecirPersona(const std::vector<double>& x, const ModeloSVM& modelo) {
    return predecirConScores(x, modelo).clase;
}

int predecirPersonaConUmbral(const std::vector<double>& x, const ModeloSVM& modelo, double umbralScoreMinimo) {
    auto p = predecirConScores(x, modelo);
    if (p.score < umbralScoreMinimo) return -1;
    return p.clase;
}

int predecirPersonaConMargen(const std::vector<double>& x, const ModeloSVM& modelo, double umbral_margen) {
    auto p = predecirConScores(x, modelo);
    const double margen = p.score - p.score2;
    if (margen < umbral_margen) return -1;
    return p.clase;
}

// ===== IMPLEMENTACIÓN QUE FALTABA (causaba el unresolved external) =====
void evaluarModeloSimple(const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const ModeloSVM& modelo,
    const std::string& nombre) {
    int aciertos = 0;
    std::map<int, int> totales, correctos;

    for (size_t i = 0; i < X.size(); ++i) {
        int pred = predecirPersona(X[i], modelo);
        if (pred == y[i]) {
            ++aciertos;
            ++correctos[y[i]];
        }
        ++totales[y[i]];
    }

    double acc = (X.empty() ? 0.0 : 100.0 * double(aciertos) / double(X.size()));
    std::cout << "\n🔎 Precisión global en " << nombre << ": " << std::fixed << std::setprecision(2) << acc << "%\n";
    std::cout << "📊 Precisión por clase:\n";
    for (auto& [clase, total] : totales) {
        double pc = (total ? 100.0 * double(correctos[clase]) / double(total) : 0.0);
        std::cout << "   Clase " << clase << ": " << correctos[clase] << "/" << total << " (" << pc << "%)\n";
    }
}

bool enTopK(const PrediccionOVA& p, int y_true, int K, const ModeloSVM& modelo) {
    // p.scores tiene tamaño = modelo.clases.size()
    std::vector<int> idx(p.scores.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Ordenar indices por score desc
    std::partial_sort(
        idx.begin(), idx.begin() + std::min(K, (int)idx.size()), idx.end(),
        [&](int a, int b){ return p.scores[a] > p.scores[b]; }
    );

    // Ver si y_true está entre las K mejores clases
    K = std::min(K, (int)idx.size());
    for (int i = 0; i < K; ++i) {
        int clase = modelo.clases[idx[i]];
        if (clase == y_true) return true;
    }
    return false;
}

int predictOVAScore(const ModeloSVM &modelo, const std::vector<double> &x,
                    double &bestScore, double &secondScore, int &bestClass)
{
    bestScore = -1e18;
    secondScore = -1e18;
    bestClass = -1;

    for (int c = 0; c < (int)modelo.clases.size(); ++c) {
        const auto &w = modelo.pesosPorClase[c];

        if (w.size() != x.size()) continue; // 🔥 mismatch => ignorar

        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) s += x[i] * w[i];
        s += modelo.biasPorClase[c];

        if (s > bestScore) {
            secondScore = bestScore;
            bestScore = s;
            bestClass = modelo.clases[c];
        } else if (s > secondScore) {
            secondScore = s;
        }
    }
    return bestClass;
}

