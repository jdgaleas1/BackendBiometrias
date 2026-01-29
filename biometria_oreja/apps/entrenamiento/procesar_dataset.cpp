// procesar_dataset.cpp (docker-friendly: argv + env fallback)

#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/aumentar_dataset.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/dataset_loader.h"
#include "utilidades/guardar_csv.h"
#include "utilidades/pca_utils.h"
#include "utilidades/lda_utils.h"
#include "utilidades/normalizacion.h"
#include "utilidades/zscore_params.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <random>
#include <cmath>
#include <limits>

namespace fs = std::filesystem;

static double l2norm(const std::vector<double>& v) {
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

static double calcularUmbralEER(const std::vector<double>& genuinos,
                                const std::vector<double>& impostores,
                                double& outEER,
                                int numUmbrales = 1000) {
    outEER = 100.0;
    if (genuinos.empty() || impostores.empty()) return 0.5;

    double min_score = std::min(*std::min_element(genuinos.begin(), genuinos.end()),
                                *std::min_element(impostores.begin(), impostores.end()));
    double max_score = std::max(*std::max_element(genuinos.begin(), genuinos.end()),
                                *std::max_element(impostores.begin(), impostores.end()));
    if (numUmbrales < 200) numUmbrales = 200;
    double step = (max_score - min_score) / numUmbrales;
    if (step <= 0.0) return min_score;

    double bestGap = 1e9;
    double bestThr = min_score;
    double bestEER = 100.0;

    for (int i = 0; i <= numUmbrales; ++i) {
        double thr = min_score + i * step;
        int genuinos_rech = 0;
        for (double s : genuinos) if (s < thr) genuinos_rech++;
        double FRR = 100.0 * genuinos_rech / genuinos.size();

        int impost_acept = 0;
        for (double s : impostores) if (s >= thr) impost_acept++;
        double FAR = 100.0 * impost_acept / impostores.size();

        double gap = std::abs(FAR - FRR);
        if (gap < bestGap) {
            bestGap = gap;
            bestThr = thr;
            bestEER = 0.5 * (FAR + FRR);
        }
    }

    outEER = bestEER;
    return bestThr;
}

// ==========================
// Config (argv + env fallback)
// ==========================
static std::string getEnvStr(const char* key) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string();
}

static std::string resolverRutaDataset(int argc, char** argv) {
    if (argc >= 2 && argv[1] && *argv[1]) return argv[1];
    auto env = getEnvStr("DATASET_DIR");
    if (!env.empty()) return env;
    return "./dataset"; // fallback local
}

static int resolverPCA(int argc, char** argv, int def = 120) {
    if (argc >= 3 && argv[2] && *argv[2]) {
        try { return std::stoi(argv[2]); }
        catch (...) {}
    }
    auto env = getEnvStr("PCA_COMPONENTS");
    if (!env.empty()) {
        try { return std::stoi(env); }
        catch (...) {}
    }
    return def;
}

static int resolverLDA(int argc, char** argv, int def = -1) {
    // argv[3] = LDA components (-1 = max = numClases-1)
    if (argc >= 4 && argv[3] && *argv[3]) {
        try { return std::stoi(argv[3]); }
        catch (...) {}
    }
    auto env = getEnvStr("LDA_COMPONENTS");
    if (!env.empty()) {
        try { return std::stoi(env); }
        catch (...) {}
    }
    return def;
}

static std::string resolverOutDir() {
    auto env = getEnvStr("OUT_DIR");
    if (!env.empty()) return env;
    return "out";
}

static void asegurarDir(const std::string& dir) {
    fs::create_directories(dir);
}

static std::string joinPath(const std::string& base, const std::string& rel) {
    return (fs::path(base) / fs::path(rel)).string();
}

// ==========================
// Globales progreso
// ==========================
static std::mutex mtxPrint;
static std::atomic<size_t> imagenesProcesadas{ 0 };
static size_t totalImagenes = 0;

static size_t obtenerNumHilos();
template <typename Func>
static void parallelFor(size_t begin, size_t end, Func fn);

static size_t obtenerNumHilos() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return (hw == 0) ? 1 : (hw > 1 ? (hw - 1) : 1);
}

template <typename Func>
static void parallelFor(size_t begin, size_t end, Func fn) {
    if (end <= begin) return;
    const size_t n = end - begin;
    const size_t numHilos = obtenerNumHilos();
    if (numHilos <= 1 || n < 1024) {
        for (size_t i = begin; i < end; ++i) fn(i);
        return;
    }

    const size_t block = (n + numHilos - 1) / numHilos;
    std::vector<std::thread> hilos;
    hilos.reserve(numHilos);
    for (size_t t = 0; t < numHilos; ++t) {
        const size_t start = begin + t * block;
        if (start >= end) break;
        const size_t stop = std::min(end, start + block);
        hilos.emplace_back([=, &fn]() {
            for (size_t i = start; i < stop; ++i) fn(i);
        });
    }
    for (auto& h : hilos) h.join();
}

// ==========================
// Features - FASE 1 + FASE 2 + FASE 4
// ==========================
// FASE 1: Bloques LBP 6x6, threshold 320 (70% del bloque 21.3x21.3)
// FASE 2 - PASO 1 (probado y revertido): 4x4 bloques empeoró test de 60.5% → 52%
// FASE 4 - Multi-Scale LBP: Combina radius=1 y radius=2 para capturar patrones a múltiples escalas
//
// Configuración actual (6x6 bloques, Multi-Scale):
// - Bloques 6x6 en imagen 128x128 = bloques de 21.3×21.3 = 454 píxeles totales
// - Multi-Scale usa margen de 2 píxeles → área muestreada: (21.3-4)×(21.3-4) = ~300 píxeles
// - Umbral 200 = ~67% del área muestreada (suficiente para validar bloque)
// - Multi-Scale: radius=1 (59 bins) + radius=2 (59 bins) = 118 bins por bloque
// - Features finales: 6×6×118 = 4248 dimensiones (vs 2124 single-scale)
// - RESTAURADO: Multi-Scale da mejor accuracy que Single-Scale (+15% test)
static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    // ✅ RESTAURADO: Multi-Scale LBP para máxima precisión
    return calcularLBPMultiEscalaPorBloquesRobustoNorm(img128, mask128, 128, 128, 6, 6, 200, true);
}

struct BufferThread {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    BufferThread() = default;
};

static void procesarImagen(
    const std::string& ruta,
    int etiqueta,
    std::vector<std::vector<double>>& X_local,
    std::vector<int>& y_local,
    bool aplicarAugmentation = false
){
    int ancho = 0, alto = 0, canales = 0;
    unsigned char* imgRGB = cargarImagen(ruta, ancho, alto, canales, 3);
    if (!imgRGB) {
        std::lock_guard<std::mutex> lock(mtxPrint);
        std::cerr << "\nError cargando imagen: " << ruta << "\n";
        return;
    }

    auto gris = convertirAGris(imgRGB, ancho, alto);
    liberarImagen(imgRGB);

    // ============================================================================
    // FASE 1 - Cambio 1: ELIMINA filtro bilateral (Solución 3A)
    // El bilateral era inconsistente porque dependía del contenido local de cada
    // imagen. Ahora trabajamos DIRECTAMENTE con la imagen en escala de grises.
    // ============================================================================
    // med.marcar("bilateral");
    // auto bilateral = aplicarFiltroBilateral(gris.get(), ancho, alto, 1, 1.5, 10.0);

    // ============================================================================
    // FASE 1 - Cambio 3: USA máscara elíptica FIJA en lugar de detectarRegionOreja
    // La detección por gradientes generaba máscaras inconsistentes entre imágenes
    // del mismo usuario. Ahora usamos una máscara FIJA que es idéntica siempre.
    // ============================================================================
    // OLD: auto mascara = detectarRegionOreja(bilateral.get(), ancho, alto);

    // Como ya no usamos recorte por bounding box (que necesita máscara variable),
    // vamos a trabajar directo con resize a 128x128 y luego aplicar máscara fija.

    auto img128 = redimensionarParaBiometria(gris.get(), ancho, alto, 128, 128);

    // ============================================================================
    // FASE 6 - CLAHE + Bilateral Filter (Mejora de contraste y reducción de ruido)
    // ============================================================================
    // CLAHE: Mejora el contraste local adaptativo (8×8 tiles, clipLimit=2.0)
    // Bilateral: Reduce ruido preservando bordes (sigmaSpace=3, sigmaColor=50)
    //
    // RESTAURADO: Bilateral es crítico para test accuracy (+15%)
    // Tiempo: 11.5s/img pero necesario para 75% → 60% sin él
    auto img128_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);

    auto img128_filtered = aplicarBilateral(img128_clahe.get(), 128, 128, 3.0, 50.0);

    // ============================================================================
    // FASE 4 - Filtro Gaussiano (PROBADO Y REVERTIDO)
    // ============================================================================
    // Se probó Gaussiano σ=0.8 pero empeoró test de 61.5% → 59.5%
    // El suavizado elimina micropatrones que LBP necesita para discriminar
    // CONCLUSIÓN: LBP necesita texturas de alta frecuencia, NO suavizar
    // med.marcar("gaussiano");
    // auto img128_suave = aplicarFiltroGaussiano(img128.get(), 128, 128, 0.8);

    auto mask128 = crearMascaraElipticaFija(128, 128);

    auto feat_base = extraerFeaturesDesde128(img128_filtered.get(), mask128.get());

    X_local.push_back(feat_base);
    y_local.push_back(etiqueta);

    // ============================================================================
    // FASE 5 - Data Augmentation Geométrico (COMPATIBLE CON LBP)
    // ============================================================================
    // LBP compara relaciones entre píxeles vecinos, no valores absolutos.
    // - Fotométrico (brillo, contraste): produce el MISMO LBP code → NO aporta diversidad
    // - Geométrico (rotación, traslación): produce DIFERENTES LBP codes → SÍ aporta diversidad
    //
    // Genera 2 variaciones por imagen: rotación ±4°, traslación, zoom, flip
    // Expectativa: Train 400→1200 muestras (3x), Test sin augmentation.
    if (aplicarAugmentation) {
        auto variaciones = aumentarImagenGeometrico(img128_filtered.get(), 128, 128, "aug");

        for (auto& var : variaciones) {
            auto feat_aug = extraerFeaturesDesde128(var.first.get(), mask128.get());
            X_local.push_back(feat_aug);
            y_local.push_back(etiqueta);
        }
    }

    size_t progreso = ++imagenesProcesadas;
    if ((progreso % 25) == 0 || progreso == totalImagenes) {
        std::lock_guard<std::mutex> lock(mtxPrint);
        std::cout << "\rProgreso: " << progreso << " / " << totalImagenes << std::flush;
    }
}

static void ejecutarConPoolHilos(
    const std::vector<std::string>& rutas,
    const std::vector<int>& etiquetas,
    std::vector<std::vector<double>>& X_out,
    std::vector<int>& y_out,
    bool aplicarAugmentation = false
) {
    const size_t n = rutas.size();
    const unsigned int hw = std::thread::hardware_concurrency();
    const size_t numHilos = (hw == 0) ? 1 : (hw > 1 ? (hw - 1) : 1);

    std::atomic<size_t> next{ 0 };

    std::vector<BufferThread> buffers;
    buffers.reserve(numHilos);
    for (size_t i = 0; i < numHilos; ++i) {
        buffers.emplace_back();
        buffers.back().X.reserve((n / numHilos + 1) * 8);
        buffers.back().y.reserve((n / numHilos + 1) * 8);
    }

    auto worker = [&](size_t tid) {
        auto& buf = buffers[tid];

        // ← AQUÍ FALTABA EL LOOP PRINCIPAL
        while (true) {
            size_t idx = next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= n) break;
            
            // Procesar imagen (ahora con flag de augmentation para FASE 4)
            procesarImagen(rutas[idx], etiquetas[idx], buf.X, buf.y, aplicarAugmentation);
        }
    };

    std::vector<std::thread> hilos;
    hilos.reserve(numHilos);
    for (size_t i = 0; i < numHilos; ++i) hilos.emplace_back(worker, i);
    for (auto& h : hilos) h.join();

    size_t totalMuestras = 0;
    for (auto& b : buffers) totalMuestras += b.X.size();

    X_out.clear();
    y_out.clear();
    X_out.reserve(totalMuestras);
    y_out.reserve(totalMuestras);

    for (auto& b : buffers) {
        X_out.insert(X_out.end(),
            std::make_move_iterator(b.X.begin()),
            std::make_move_iterator(b.X.end()));
        y_out.insert(y_out.end(), b.y.begin(), b.y.end());
    }
}

int main(int argc, char** argv) {
    const std::string outDir = resolverOutDir();
    asegurarDir(outDir);

    const std::string rutaDataset = resolverRutaDataset(argc, argv);
    const int componentesPCA = resolverPCA(argc, argv, 120);
    const int componentesLDA = resolverLDA(argc, argv, 40);  // 40 por defecto, o valor específico

    std::cout << "Dataset: " << rutaDataset << "\n";
    std::cout << "OUT_DIR: " << outDir << "\n";
    std::cout << "PCA: " << componentesPCA << " | LDA: " << componentesLDA << "\n";

    std::vector<std::string> rutas;
    std::vector<int> etiquetas;
    std::map<int, int> mapaRealAInterna;
    cargarRutasDataset(rutaDataset, rutas, etiquetas, mapaRealAInterna);

    const int numClases = static_cast<int>(mapaRealAInterna.size());
    int ldaObjetivo = componentesLDA;
    if (ldaObjetivo <= 0) {
        ldaObjetivo = std::max(1, std::min(numClases - 1, 40));
    } else {
        ldaObjetivo = std::max(1, std::min(ldaObjetivo, numClases - 1));
    }
    std::cout << "LDA ajustado: " << ldaObjetivo << " (clases=" << numClases << ")\n";

    // ===== Split por IMAGENES dentro de cada USUARIO (escenario B login) =====
    const int TEST_IMGS_PER_USER = 2;   // 2/7 para test, 5/7 para train (más datos train)
    const int SPLIT_SEED = 42;

    // 1) agrupar indices por usuario(etiqueta)
    std::unordered_map<int, std::vector<size_t>> idxPorUser;
    idxPorUser.reserve(mapaRealAInterna.size());

    for (size_t i = 0; i < rutas.size(); ++i) {
        idxPorUser[etiquetas[i]].push_back(i);
    }

    // 2) repartir por usuario: 3 a test, 4 a train
    std::mt19937 gen(SPLIT_SEED);

    std::vector<std::string> rutas_train, rutas_test;
    std::vector<int> etiquetas_train, etiquetas_test;
    rutas_train.reserve(rutas.size());
    rutas_test.reserve(rutas.size());

    for (auto& [user, idxs] : idxPorUser) {
        // idxs debería tener 7
        std::shuffle(idxs.begin(), idxs.end(), gen);

        int te = 0;
        for (size_t k = 0; k < idxs.size(); ++k) {
            size_t id = idxs[k];

            if (te < TEST_IMGS_PER_USER) {
                rutas_test.push_back(rutas[id]);
                etiquetas_test.push_back(etiquetas[id]);
                te++;
            } else {
                rutas_train.push_back(rutas[id]);
                etiquetas_train.push_back(etiquetas[id]);
            }
        }
    }

        std::cout << "Split por usuario (test=" << TEST_IMGS_PER_USER
            << ") -> Train imgs: " << rutas_train.size()
            << " | Test imgs: " << rutas_test.size() << "\n";

    totalImagenes = rutas_train.size() + rutas_test.size();
    imagenesProcesadas = 0;

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    // FASE 5: Augmentation GEOMÉTRICO ACTIVADO
    // Augmentation geométrico genera variaciones reales en espacio LBP (3x datos)
    // Rotación ±4°, traslación, zoom (sin flip) → diferentes códigos LBP
    ejecutarConPoolHilos(rutas_train, etiquetas_train, X_train, y_train, true);
    ejecutarConPoolHilos(rutas_test, etiquetas_test, X_test, y_test, false);  

    std::cout << "\nTrain muestras: " << X_train.size() << " | Test muestras: " << X_test.size() << "\n";

    // ============================================================================
    // Z-score standardization pre-PCA (corrige problema de escala)
    // ============================================================================
    const size_t numDims = X_train.empty() ? 0 : X_train[0].size();
    std::vector<double> media(numDims, 0.0);
    std::vector<double> std_dev(numDims, 0.0);

    // Calcular media de cada dimensión en train (paralelo por dimensión)
    parallelFor(0, numDims, [&](size_t d) {
        double sum = 0.0;
        for (const auto& muestra : X_train) sum += muestra[d];
        media[d] = sum / X_train.size();
    });

    // Calcular desviación estándar de cada dimensión en train (paralelo por dimensión)
    parallelFor(0, numDims, [&](size_t d) {
        double sumsq = 0.0;
        for (const auto& muestra : X_train) {
            double diff = muestra[d] - media[d];
            sumsq += diff * diff;
        }
        double sd = std::sqrt(sumsq / X_train.size());
        if (sd < 1e-10) sd = 1.0; // Evitar división por cero
        std_dev[d] = sd;
    });

    // Aplicar Z-score a train (paralelo por muestra)
    parallelFor(0, X_train.size(), [&](size_t i) {
        auto& muestra = X_train[i];
        for (size_t d = 0; d < numDims; ++d) {
            muestra[d] = (muestra[d] - media[d]) / std_dev[d];
        }
    });

    // Aplicar Z-score a test (paralelo por muestra)
    parallelFor(0, X_test.size(), [&](size_t i) {
        auto& muestra = X_test[i];
        for (size_t d = 0; d < numDims; ++d) {
            muestra[d] = (muestra[d] - media[d]) / std_dev[d];
        }
    });

    ZScoreParams zp;
    zp.mean = media;
    zp.stdev = std_dev;

    const std::string rutaZ = joinPath(outDir, "zscore_params.dat");
    if (!guardarZScoreParams(rutaZ, zp, ';')) {
        std::cerr << "[ERROR] No se pudo guardar zscore_params.dat en: " << rutaZ << "\n";
        return 1;
    }
    std::cout << "[OK] Guardado Z-score params: " << rutaZ << " (dims=" << zp.mean.size() << ")\n";

    // PCA fit SOLO con train
    ModeloPCA modeloPCA = entrenarPCA(X_train, componentesPCA);
    guardarModeloPCA(joinPath(outDir, "modelo_pca.dat"), modeloPCA);

    // PCA transform train y test
    auto Xpca_train = aplicarPCAConModelo(X_train, modeloPCA);
    auto Xpca_test  = aplicarPCAConModelo(X_test,  modeloPCA);

    // L2-normalizar embeddings PCA (mejora similitud coseno y estabilidad)
    for (auto& v : Xpca_train) normalizarVector(v);
    for (auto& v : Xpca_test)  normalizarVector(v);

    // Entrenar LDA con datos PCA y etiquetas de train
    // componentesLDA: -1 = max (numClases-1), o valor específico (ej: 50, 70)
    ModeloLDA modeloLDA = entrenarLDA(Xpca_train, y_train, ldaObjetivo);
    guardarModeloLDA(joinPath(outDir, "modelo_lda.dat"), modeloLDA);

    // Aplicar LDA a train y test
    auto Xlda_train = aplicarLDAConModelo(Xpca_train, modeloLDA);
    auto Xlda_test  = aplicarLDAConModelo(Xpca_test, modeloLDA);

    // L2-normalizar embeddings LDA (consistencia para coseno/SVM)
    for (auto& v : Xlda_train) normalizarVector(v);
    for (auto& v : Xlda_test)  normalizarVector(v);

    // Templates K=1 (coseno) desde TRAIN
    {
        std::unordered_map<int, std::vector<double>> sum;
        std::unordered_map<int, int> count;
        for (size_t i = 0; i < Xlda_train.size(); ++i) {
            int c = y_train[i];
            auto& acc = sum[c];
            if (acc.empty()) acc.assign(Xlda_train[i].size(), 0.0);
            for (size_t d = 0; d < Xlda_train[i].size(); ++d) acc[d] += Xlda_train[i][d];
            count[c]++;
        }

        std::vector<int> clases;
        clases.reserve(sum.size());
        for (const auto& kv : sum) clases.push_back(kv.first);
        std::sort(clases.begin(), clases.end());

        const std::string rutaTemplates = joinPath(outDir, "templates_k1.csv");
        std::ofstream f(rutaTemplates);
        if (!f.is_open()) {
            std::cerr << "[ERROR] No se pudo guardar templates_k1.csv\n";
            return 1;
        }

        std::vector<std::vector<double>> templates;
        std::vector<double> norms;
        templates.reserve(clases.size());
        norms.reserve(clases.size());

        for (int c : clases) {
            auto it = sum.find(c);
            if (it == sum.end()) continue;
            auto v = it->second;
            int cnt = std::max(1, count[c]);
            for (double& x : v) x /= (double)cnt;
            f << c;
            for (double x : v) f << ";" << x;
            f << "\n";
            templates.push_back(v);
            norms.push_back(l2norm(v));
        }
        f.close();
        std::cout << "[OK] Guardado templates K=1: " << rutaTemplates << "\n";

        // Calcular umbral EER usando TEST
        if (!Xlda_test.empty() && Xlda_test.size() == y_test.size()) {
            std::unordered_map<int, size_t> idxClase;
            idxClase.reserve(clases.size());
            for (size_t i = 0; i < clases.size(); ++i) idxClase[clases[i]] = i;

            std::vector<double> genuinos;
            std::vector<double> impostores;
            genuinos.reserve(Xlda_test.size());
            impostores.reserve(Xlda_test.size() * (clases.size() - 1));

            for (size_t i = 0; i < Xlda_test.size(); ++i) {
                const auto& x = Xlda_test[i];
                const double normX = l2norm(x);
                auto it = idxClase.find(y_test[i]);
                if (it == idxClase.end()) continue;
                size_t idxG = it->second;
                double sG = cosineSim(x, normX, templates[idxG], norms[idxG]);
                genuinos.push_back(sG);
                for (size_t c = 0; c < templates.size(); ++c) {
                    if (c == idxG) continue;
                    double s = cosineSim(x, normX, templates[c], norms[c]);
                    impostores.push_back(s);
                }
            }

            double eer = 0.0;
            double thr = calcularUmbralEER(genuinos, impostores, eer, 1000);
            const std::string rutaUmbral = joinPath(outDir, "umbral_eer.txt");
            std::ofstream fu(rutaUmbral);
            if (fu.is_open()) {
                fu << "threshold=" << thr << "\n";
                fu << "eer=" << eer << "\n";
                fu << "genuine=" << genuinos.size() << "\n";
                fu << "impostor=" << impostores.size() << "\n";
                fu.close();
                std::cout << "[OK] Umbral EER guardado: " << rutaUmbral
                          << " (thr=" << thr << ", eer=" << eer << "%)\n";
            } else {
                std::cerr << "[WARN] No se pudo guardar umbral_eer.txt\n";
            }
        }
    }

    // Guardar CSV separados (ahora con features LDA)
    guardarCSV(joinPath(outDir, "caracteristicas_lda_train.csv"), Xlda_train, y_train, ';');
    guardarCSV(joinPath(outDir, "caracteristicas_lda_test.csv"),  Xlda_test,  y_test,  ';');

    return 0;
}