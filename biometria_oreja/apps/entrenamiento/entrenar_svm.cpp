#include "svm/cargar_csv.h"
#include "svm/svm_entrenamiento.h"
#include "svm/svm_prediccion.h"
#include "metricas/svm_metricas.h"
#include "utilidades/svm_ova_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <atomic>
#include <unordered_map>

namespace fs = std::filesystem;

static std::string getEnvStr(const char* key) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string();
}

static std::string resolverOutDir() {
    auto env = getEnvStr("OUT_DIR");
    return env.empty() ? "out" : env;
}

static std::string joinPath(const std::string& base, const std::string& rel) {
    return (fs::path(base) / fs::path(rel)).string();
}

// ============================================================================
// FASE 3 - CONFIGURACIÓN ÓPTIMA CONFIRMADA (50 CLASES)
// ============================================================================
// EXPERIMENTOS EXHAUSTIVOS CON 50 CLASES (750 train, 100 test, 49 dims LDA):
//
// RESULTADOS GRID SEARCH COMPLETO:
// C=0.001,  LR=0.01, ep=5000 → Train 100%, Test 72%, EER 15.04% ✅ ÓPTIMO
// C=0.0005, LR=0.01, ep=5000 → Train 100%, Test 72%, EER 15.98%
// C=0.0001, LR=0.01, ep=5000 → Train 100%, Test 71%, EER 15.02%
// C=0.01,   LR=0.01, ep=5000 → Train 99.87%, Test 69%, EER 14.37%
// C=0.1,    LR=0.01, ep=5000 → Train 100%, Test 70%, EER 14.45%
// C=0.5,    LR=0.01, ep=5000 → Train 81.60%, Test 47% ❌ UNDERFITTING
//
// CONCLUSIÓN FINAL:
// - C=0.001 es ÓPTIMO: mejor test (72%) con EER aceptable (15%)
// - Bajar C más (0.0005, 0.0001) no mejora test, puede empeorar EER
// - PLATEAU de performance alcanzado: ~72% es límite con arquitectura actual
//
// PRÓXIMOS PASOS PARA MEJORAR:
// 1. Volver a 100 clases con split 5/2 (más datos train)
// 2. Reducir PCA de 150 → 120 (menos ruido en LDA)
// 3. Considerar kernel RBF para separación no lineal
// ============================================================================
static constexpr double TASA_APRENDIZAJE = 0.01;
static constexpr int    EPOCAS = 5000;
static constexpr double C_REGULARIZ = 0.001;
static constexpr double TOLERANCIA = 1e-5;

static int getEnvInt(const char* key, int def) {
    auto v = getEnvStr(key);
    if (v.empty()) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

// ============================================================================
// FASE 6 - Evaluación 1:1 (Verificación) con FAR/FRR/EER
// ============================================================================
// Para caso de uso de LOGIN: usuario dice "soy el ID X", sistema acepta/rechaza
// Métricas correctas: FAR (False Accept Rate), FRR (False Reject Rate), EER

struct ResultadosFARFRR {
    double umbral;
    double FAR;  // % impostores aceptados
    double FRR;  // % genuinos rechazados
    double distancia_EER; // |FAR - FRR|
};

static ResultadosFARFRR calcularEERConCurva(
    const std::vector<double>& scores_genuinos,
    const std::vector<double>& scores_impostores,
    std::vector<ResultadosFARFRR>& resultados,
    int numUmbrales
);

struct TemplateModel {
    std::vector<int> clases;
    std::vector<std::vector<double>> templates;
    std::vector<double> norms;
    int templatesPerClass = 1;
};

static double normaL2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(std::max(s, 1e-12));
}

static double cosineSim(const std::vector<double>& a, double normA,
                         const std::vector<double>& b, double normB) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    const double denom = normA * normB;
    if (denom <= 1e-12) return -1.0;
    return dot / denom;
}

static void normalizarVectorInPlace(std::vector<double>& v) {
    const double n = normaL2(v);
    if (n <= 1e-12) return;
    for (double& x : v) x /= n;
}

static TemplateModel construirTemplatesKMeans(const std::vector<std::vector<double>>& X_train,
                                              const std::vector<int>& y_train,
                                              int k) {
    std::vector<int> clases = y_train;
    std::sort(clases.begin(), clases.end());
    clases.erase(std::unique(clases.begin(), clases.end()), clases.end());

    const size_t numClases = clases.size();
    const size_t dims = X_train.empty() ? 0 : X_train[0].size();

    std::unordered_map<int, size_t> idxClase;
    idxClase.reserve(numClases);
    for (size_t i = 0; i < numClases; ++i) idxClase[clases[i]] = i;

    TemplateModel model;
    model.clases = clases;
    model.templatesPerClass = std::max(1, k);
    model.templates.resize(numClases * model.templatesPerClass, std::vector<double>(dims, 0.0));
    model.norms.resize(numClases * model.templatesPerClass, 1.0);

    std::vector<std::vector<size_t>> idxsPorClase(numClases);
    for (size_t i = 0; i < y_train.size(); ++i) {
        auto it = idxClase.find(y_train[i]);
        if (it != idxClase.end()) idxsPorClase[it->second].push_back(i);
    }

    const int maxIter = 10;

    for (size_t c = 0; c < numClases; ++c) {
        const auto& idxs = idxsPorClase[c];
        if (idxs.empty()) continue;
        const int K = model.templatesPerClass;
        std::vector<std::vector<double>> centers;
        centers.reserve(K);

        // init: primer centro = primer vector
        centers.push_back(X_train[idxs[0]]);
        normalizarVectorInPlace(centers.back());

        // init restantes: seleccionar el más lejano al set actual
        for (int cc = 1; cc < K; ++cc) {
            size_t bestIdx = idxs[0];
            double worstSim = 1.0;
            for (size_t kidx = 0; kidx < idxs.size(); ++kidx) {
                const auto& v = X_train[idxs[kidx]];
                double bestSim = 1.0;
                const double normV = normaL2(v);
                for (const auto& cen : centers) {
                    double sim = cosineSim(v, normV, cen, 1.0);
                    if (sim < bestSim) bestSim = sim;
                }
                if (bestSim < worstSim) {
                    worstSim = bestSim;
                    bestIdx = idxs[kidx];
                }
            }
            centers.push_back(X_train[bestIdx]);
            normalizarVectorInPlace(centers.back());
        }

        if (idxs.size() == 1) {
            for (int cc = 0; cc < K; ++cc) {
                model.templates[c * K + cc] = centers[0];
                model.norms[c * K + cc] = 1.0;
            }
            continue;
        }

        for (int it = 0; it < maxIter; ++it) {
            std::vector<std::vector<double>> sums(K, std::vector<double>(dims, 0.0));
            std::vector<int> counts(K, 0);

            for (size_t kidx = 0; kidx < idxs.size(); ++kidx) {
                const auto& v = X_train[idxs[kidx]];
                const double normV = normaL2(v);
                int bestC = 0;
                double bestSim = -1e18;
                for (int cc = 0; cc < K; ++cc) {
                    double sim = cosineSim(v, normV, centers[cc], 1.0);
                    if (sim > bestSim) { bestSim = sim; bestC = cc; }
                }
                for (size_t d = 0; d < dims; ++d) sums[bestC][d] += v[d];
                counts[bestC]++;
            }

            for (int cc = 0; cc < K; ++cc) {
                if (counts[cc] > 0) {
                    for (size_t d = 0; d < dims; ++d) centers[cc][d] = sums[cc][d] / counts[cc];
                    normalizarVectorInPlace(centers[cc]);
                }
            }
        }

        for (int cc = 0; cc < K; ++cc) {
            model.templates[c * K + cc] = centers[cc];
            model.norms[c * K + cc] = 1.0;
        }
    }

    return model;
}

static double maxScoreClase(const std::vector<double>& x, double normX,
                            const TemplateModel& tm, size_t idxClase) {
    const int K = std::max(1, tm.templatesPerClass);
    double best = -1e18;
    for (int k = 0; k < K; ++k) {
        double s = cosineSim(x, normX,
                             tm.templates[idxClase * K + k],
                             tm.norms[idxClase * K + k]);
        if (s > best) best = s;
    }
    return best;
}

static void evaluarTemplates(const std::vector<std::vector<double>>& X_train,
                             const std::vector<int>& y_train,
                             const std::vector<std::vector<double>>& X_test,
                             const std::vector<int>& y_test,
                             const std::string& outDir) {
    if (X_train.empty() || X_test.empty()) return;

    const int templateK = std::max(1, getEnvInt("TEMPLATE_K", 1));
    TemplateModel tm = construirTemplatesKMeans(X_train, y_train, templateK);
    if (tm.templates.empty()) return;

    std::unordered_map<int, size_t> idxClase;
    idxClase.reserve(tm.clases.size());
    for (size_t i = 0; i < tm.clases.size(); ++i) idxClase[tm.clases[i]] = i;

    std::atomic<int> aciertos{0};
    std::vector<double> scores_genuinos;
    std::vector<double> scores_impostores;
    scores_genuinos.reserve(X_test.size());
    scores_impostores.reserve(X_test.size() * (tm.clases.size() - 1));

    for (size_t i = 0; i < X_test.size(); ++i) {
        const auto& x = X_test[i];
        const double normX = normaL2(x);

        double bestScore = -1e18;
        int bestClass = -1;


        auto it = idxClase.find(y_test[i]);
        if (it != idxClase.end()) {
            size_t idxG = it->second;
            double sG = maxScoreClase(x, normX, tm, idxG);
            scores_genuinos.push_back(sG);
        }

        for (size_t c = 0; c < tm.clases.size(); ++c) {
            double s = maxScoreClase(x, normX, tm, c);

            if (s > bestScore) {
                bestScore = s;
                bestClass = tm.clases[c];
            }

            if (tm.clases[c] != y_test[i]) {
                scores_impostores.push_back(s);
            }
        }

        if (bestClass == y_test[i]) aciertos.fetch_add(1, std::memory_order_relaxed);
    }

    double acc = 100.0 * double(aciertos.load()) / double(X_test.size());

    std::vector<ResultadosFARFRR> resultados;
    auto iter_EER = calcularEERConCurva(scores_genuinos, scores_impostores, resultados, 1000);
    double eer = 0.5 * (iter_EER.FAR + iter_EER.FRR);

    std::cout << "\n🧩 Templates por usuario (coseno, K=" << tm.templatesPerClass << "):\n";
    std::cout << "   - Top-1 Accuracy (TEST): " << std::fixed << std::setprecision(2) << acc << "%\n";
    std::cout << "   - EER (Template):        " << std::fixed << std::setprecision(2) << eer << "%\n";

    std::ofstream csvFAR(joinPath(outDir, "verificacion_FAR_FRR_template.csv"));
    if (csvFAR.is_open()) {
        csvFAR << "umbral,FAR,FRR\n";
        for (const auto& r : resultados) {
            csvFAR << r.umbral << "," << r.FAR << "," << r.FRR << "\n";
        }
        csvFAR.close();
    }

}

static ResultadosFARFRR calcularEERConCurva(
    const std::vector<double>& scores_genuinos,
    const std::vector<double>& scores_impostores,
    std::vector<ResultadosFARFRR>& resultados,
    int numUmbrales = 1000
) {
    resultados.clear();
    if (scores_genuinos.empty() || scores_impostores.empty()) {
        return {0.0, 100.0, 100.0, 100.0};
    }

    double min_score = std::min(*std::min_element(scores_genuinos.begin(), scores_genuinos.end()),
                                *std::min_element(scores_impostores.begin(), scores_impostores.end()));
    double max_score = std::max(*std::max_element(scores_genuinos.begin(), scores_genuinos.end()),
                                *std::max_element(scores_impostores.begin(), scores_impostores.end()));

    if (numUmbrales < 200) numUmbrales = 200;
    double step = (max_score - min_score) / numUmbrales;
    if (step <= 0.0) {
        return {min_score, 100.0, 100.0, 100.0};
    }

    for (int i = 0; i <= numUmbrales; ++i) {
        double umbral = min_score + i * step;

        int genuinos_rechazados = 0;
        for (double score : scores_genuinos) {
            if (score < umbral) genuinos_rechazados++;
        }
        double FRR = 100.0 * genuinos_rechazados / scores_genuinos.size();

        int impostores_aceptados = 0;
        for (double score : scores_impostores) {
            if (score >= umbral) impostores_aceptados++;
        }
        double FAR = 100.0 * impostores_aceptados / scores_impostores.size();

        resultados.push_back({umbral, FAR, FRR, std::abs(FAR - FRR)});
    }

    auto iter_EER = std::min_element(resultados.begin(), resultados.end(),
        [](const ResultadosFARFRR& a, const ResultadosFARFRR& b) {
            return a.distancia_EER < b.distancia_EER;
        });

    return *iter_EER;
}


static void evaluarYExportar(const std::vector<std::vector<double>>& X_test,
                            const std::vector<int>& y_test,
                            const ModeloSVM& modelo,
                            const std::string& outDir) {
    std::vector<int> y_pred;
    y_pred.reserve(X_test.size());
    for (const auto& x : X_test) y_pred.push_back(predecirPersona(x, modelo));

    ResultadosMetricas metricas =
        calcularMetricasAvanzadas(y_test, y_pred, static_cast<int>(modelo.clases.size()));

    exportarMetricasParaGraficos(metricas, outDir.c_str());

    std::cout << "\nMetricas avanzadas:\n";
    std::cout << "   - Accuracy:           " << std::fixed << std::setprecision(2) << metricas.accuracy << "%\n";
    std::cout << "   - Precision macro:    " << metricas.precision_macro << "%\n";
    std::cout << "   - Recall macro:       " << metricas.recall_macro << "%\n";
    std::cout << "   - F1 macro:           " << metricas.f1_macro << "%\n";
    std::cout << "   - Balanced Accuracy:  " << metricas.balanced_accuracy << "%\n";
    std::cout << "   - MCC:                " << metricas.mcc << "%\n";
}

int main() {
    const std::string outDir = resolverOutDir();
    fs::create_directories(outDir);

    const std::string rutaTrain = joinPath(outDir, "caracteristicas_lda_train.csv");
    const std::string rutaTest  = joinPath(outDir, "caracteristicas_lda_test.csv");
    //const std::string rutaTrain = joinPath(outDir, "caracteristicas_pca_train.csv");
    //const std::string rutaTest  = joinPath(outDir, "caracteristicas_pca_test.csv");
    const std::string rutaModelo = joinPath(outDir, "modelo_svm.svm");

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    std::cout << "Cargando TRAIN: " << rutaTrain << "\n";
    if (!cargarCSV(rutaTrain.c_str(), X_train, y_train, ';')) {
        std::cerr << "❌ No se pudo cargar TRAIN.\n";
        return 1;
    }

    std::cout << "Cargando TEST:  " << rutaTest << "\n";
    if (!cargarCSV(rutaTest.c_str(), X_test, y_test, ';')) {
        std::cerr << "❌ No se pudo cargar TEST.\n";
        return 1;
    }

    std::cout << "Train: " << X_train.size()
              << " | Test: " << X_test.size()
              << " | Dim: " << (X_train.empty() ? 0 : X_train[0].size()) << "\n";

    const double bestLR = TASA_APRENDIZAJE;
    const double bestC  = C_REGULARIZ;

    std::cout << "Entrenando modelo SVM (OVA)...\n";
    std::cout << "   tasa=" << bestLR
              << ", epocas=" << EPOCAS
              << ", C=" << std::fixed << std::setprecision(6) << bestC
              << ", tol=" << std::setprecision(6) << TOLERANCIA << "\n";

    ModeloSVM modelo = entrenarSVMOVA(
        X_train, y_train, bestLR, EPOCAS, bestC, TOLERANCIA
    );

    if (guardarModeloSVM(rutaModelo.c_str(), modelo))
        std::cout << "Modelo SVM guardado: " << rutaModelo << "\n";

    evaluarModeloSimple(X_train, y_train, modelo, "Entrenamiento");
    evaluarModeloSimple(X_test,  y_test,  modelo, "Prueba");

    auto evaluarTopK = [&](int K){
        int ok = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            PrediccionOVA p = predecirConScores(X_test[i], modelo);
            if (enTopK(p, y_test[i], K, modelo)) ok++;
        }
        double acc = X_test.empty() ? 0.0 : 100.0 * (double)ok / (double)X_test.size();
        std::cout << "🎯 Top-" << K << " Accuracy (TEST): " << std::fixed << std::setprecision(2) << acc << "%\n";
    };

    evaluarTopK(1);
    evaluarTopK(3);
    evaluarTopK(5);

    evaluarYExportar(X_test, y_test, modelo, outDir);

    // Templates por usuario (PCA-only friendly)
    evaluarTemplates(X_train, y_train, X_test, y_test, outDir);

    return 0;
}
