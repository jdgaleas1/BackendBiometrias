#include "svm/cargar_csv.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <cstdlib>

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

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    const int n = std::min((int)a.size(), (int)b.size());
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static double norm2(const std::vector<double>& a) {
    double s = 0.0;
    for (double v : a) s += v*v;
    return std::sqrt(s);
}

static double cosineSim(const std::vector<double>& a, const std::vector<double>& b) {
    const double na = norm2(a);
    const double nb = norm2(b);
    if (na < 1e-12 || nb < 1e-12) return -1e18;
    return dot(a,b) / (na*nb);
}

int main() {
    const std::string outDir = resolverOutDir();
    const std::string rutaTrain = joinPath(outDir, "caracteristicas_lda_train.csv");
    const std::string rutaTest  = joinPath(outDir, "caracteristicas_lda_test.csv");

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

    if (X_train.empty() || X_test.empty()) {
        std::cerr << "❌ CSV vacío.\n";
        return 1;
    }

    const int dim = (int)X_train[0].size();
    std::cout << "Dim: " << dim << " | Train: " << X_train.size() << " | Test: " << X_test.size() << "\n";

    // ===== 1) construir centroides por clase =====
    std::unordered_map<int, std::vector<double>> suma;
    std::unordered_map<int, int> conteo;
    suma.reserve(256);
    conteo.reserve(256);

    for (size_t i = 0; i < X_train.size(); ++i) {
        int c = y_train[i];
        if (suma.find(c) == suma.end()) suma[c] = std::vector<double>(dim, 0.0);
        auto& acc = suma[c];
        for (int j = 0; j < dim; ++j) acc[j] += X_train[i][j];
        conteo[c]++;
    }

    std::vector<int> clases;
    clases.reserve(suma.size());

    std::vector<std::vector<double>> centroides;
    centroides.reserve(suma.size());

    for (auto& kv : suma) {
        int c = kv.first;
        auto v = kv.second;
        int n = conteo[c];
        for (int j = 0; j < dim; ++j) v[j] /= std::max(1, n);
        clases.push_back(c);
        centroides.push_back(std::move(v));
    }

    // ===== 2) predicción por cosine a centroides =====
    auto topK = [&](const std::vector<double>& x, int K) {
        std::vector<std::pair<double,int>> scores;
        scores.reserve(centroides.size());
        for (size_t i = 0; i < centroides.size(); ++i) {
            double s = cosineSim(x, centroides[i]);
            scores.push_back({s, clases[i]});
        }
        K = std::min(K, (int)scores.size());
        std::partial_sort(scores.begin(), scores.begin()+K, scores.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        scores.resize(K);
        return scores; // (score, clase)
    };

    auto evalTopK = [&](int K) {
        int ok = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            auto tk = topK(X_test[i], K);
            for (auto& p : tk) {
                if (p.second == y_test[i]) { ok++; break; }
            }
        }
        double acc = X_test.empty() ? 0.0 : 100.0 * (double)ok / (double)X_test.size();
        std::cout << "🎯 Template+Cosine Top-" << K << " (TEST): " << acc << "%\n";
    };

    evalTopK(1);
    evalTopK(3);
    evalTopK(5);

    return 0;
}
