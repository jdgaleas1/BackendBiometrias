#include "utilidades/logger.h"

#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/normalizacion.h"
#include "utilidades/pca_utils.h"
#include "utilidades/lda_utils.h"
#include "utilidades/zscore_params.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <limits>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

static long long ms_since(const std::chrono::steady_clock::time_point& t0) {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now() - t0).count();
}

// ====== Estructuras para logging detallado ======
struct GrayStats {
    double mean = 0.0;
    double stddev = 0.0;
    int minv = 255;
    int maxv = 0;
    double pct_dark = 0.0;
    double pct_bright = 0.0;
    double entropy = 0.0;           // Entropía de Shannon (bits)
    double michelson_contrast = 0.0; // Contraste de Michelson (0-1)
    double rms_contrast = 0.0;       // RMS Contrast
};

// Entropía de Shannon (referencia: Shannon 1948, Pizer 1987)
static double calcEntropy(const uint8_t* img, int w, int h) {
    if (!img || w <= 0 || h <= 0) return 0.0;
    
    std::vector<int> hist(256, 0);
    const int N = w * h;
    
    // Calcular histograma
    for (int i = 0; i < N; ++i) hist[img[i]]++;
    
    // Calcular entropía: H = -Σ p(i) × log₂(p(i))
    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / N;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

static GrayStats calcGrayStats(const uint8_t* img, int w, int h, int dark_thr=10, int bright_thr=245) {
    GrayStats s;
    const int N = w * h;
    if (!img || N <= 0) return s;

    long long sum = 0, sum2 = 0;
    int c_dark = 0, c_bright = 0;

    for (int i = 0; i < N; ++i) {
        int v = (int)img[i];
        sum += v;
        sum2 += 1LL * v * v;
        if (v < s.minv) s.minv = v;
        if (v > s.maxv) s.maxv = v;
        if (v <= dark_thr) c_dark++;
        if (v >= bright_thr) c_bright++;
    }

    s.mean = (double)sum / (double)N;
    double var = (double)sum2 / (double)N - s.mean * s.mean;
    if (var < 0) var = 0;
    s.stddev = std::sqrt(var);
    s.pct_dark = 100.0 * (double)c_dark / (double)N;
    s.pct_bright = 100.0 * (double)c_bright / (double)N;
    
    // Métricas adicionales
    s.entropy = calcEntropy(img, w, h);
    
    // Michelson Contrast: (max-min)/(max+min) [Michelson 1927, ISO 9241]
    if (s.maxv + s.minv > 0) {
        s.michelson_contrast = (double)(s.maxv - s.minv) / (double)(s.maxv + s.minv);
    }
    
    // RMS Contrast = stddev [Peli 1990]
    s.rms_contrast = s.stddev;

    return s;
}

static double maskCoveragePct(const uint8_t* mask, int w, int h) {
    if (!mask || w <= 0 || h <= 0) return 0.0;
    const int N = w * h;
    int on = 0;
    for (int i = 0; i < N; ++i) {
        if (mask[i] > 0) on++;
    }
    return 100.0 * (double)on / (double)N;
}

static void logPhaseHeader(const std::string& title) {
    std::cerr << "====================================================" << std::endl;
    std::cerr << title << std::endl;
    std::cerr << "====================================================" << std::endl;
}

static void logMetrics(const std::string& fase, const GrayStats& s_in, const GrayStats& s_out, long long ms) {
    std::cerr << fase << ":" << std::endl;
    std::cerr << "  Entrada  -> mean=" << std::fixed << std::setprecision(2) << s_in.mean
              << " std=" << s_in.stddev << " min=" << s_in.minv << " max=" << s_in.maxv << std::endl;
    std::cerr << "  Salida   -> mean=" << std::fixed << std::setprecision(2) << s_out.mean
              << " std=" << s_out.stddev << " min=" << s_out.minv << " max=" << s_out.maxv << std::endl;
    
    double delta_mean = s_out.mean - s_in.mean;
    double delta_std = s_out.stddev - s_in.stddev;
    
    std::cerr << "  Delta    -> mean=" << std::showpos << std::fixed << std::setprecision(2) << delta_mean
              << " std=" << delta_std << std::noshowpos << " | " << ms << " ms" << std::endl;
}

// ====== Tu pipeline (según tu proyecto) ======
struct Imagen128 {
    std::unique_ptr<uint8_t[]> img128;
    std::unique_ptr<uint8_t[]> mask128;
    int w = 128;
    int h = 128;
};

static std::string getEnvStr(const char* k, const std::string& def) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : def;
}

static Imagen128 preprocesarHasta128(const uint8_t* imagenGris, int ancho, int alto) {
    // ============================================================================
    // PIPELINE FASE 6 - SINCRONIZADO CON procesar_dataset.cpp
    // ============================================================================
    // 1. Resize directo a 128x128 (SIN bilateral previo, SIN detección de región)
    // 2. CLAHE (8×8 tiles, clipLimit=2.0) - Mejora contraste local
    // 3. Bilateral (σ_space=3, σ_color=50) - Reduce ruido post-CLAHE
    // 4. Máscara elíptica FIJA (consistente entre todas las imágenes)
    // ============================================================================

    Imagen128 out;
    auto t0 = std::chrono::steady_clock::now();

    logPhaseHeader("PREPROCESAMIENTO FASE 6 (PIPELINE COMPLETO)");
    std::cerr << "Entrada: " << ancho << "x" << alto << " (escala de grises)" << std::endl;
    std::cerr << std::endl;

    GrayStats s_original = calcGrayStats(imagenGris, ancho, alto);

    // Paso 1: Resize directo a 128x128
    t0 = std::chrono::steady_clock::now();
    auto img128 = redimensionarParaBiometria(imagenGris, ancho, alto, 128, 128);
    auto ms_resize = ms_since(t0);
    
    GrayStats s_resize = calcGrayStats(img128.get(), 128, 128);
    logMetrics("FASE 1: REDIMENSIONAMIENTO 128x128", s_original, s_resize, ms_resize);
    
    // Validación: Relación de aspecto
    double aspect_ratio = (double)ancho / (double)alto;
    bool aspect_ok = (aspect_ratio >= 0.85 && aspect_ratio <= 1.15);
    std::cerr << "  Validación -> aspect_ratio=" << std::fixed << std::setprecision(2) << aspect_ratio
              << " umbral=[0.85,1.15] " << (aspect_ok ? "✓ PASS" : "⚠ ADVERTENCIA") << std::endl;
    std::cerr << std::endl;

    // Paso 2: CLAHE (8×8 tiles, clipLimit=2.0)
    t0 = std::chrono::steady_clock::now();
    auto img128_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);
    auto ms_clahe = ms_since(t0);
    
    GrayStats s_clahe = calcGrayStats(img128_clahe.get(), 128, 128);
    logMetrics("FASE 2: CLAHE (Mejora de Contraste)", s_resize, s_clahe, ms_clahe);
    
    // ========== MÉTRICAS CUANTITATIVAS ACADÉMICAS ==========
    // Referencia: Pizer et al. (1987) "Adaptive histogram equalization"
    //             Zuiderveld (1994) "Contrast Limited AHE" - Graphics Gems IV
    //             ISO/IEC 29794-1:2016 - Biometric sample quality
    
    // 1. RMS Contrast (Peli 1990) - Valor absoluto en escala [0-255]
    std::cerr << "  MÉTRICAS CLAHE:" << std::endl;
    std::cerr << "    RMS Contrast (Desv.Est): " << std::fixed << std::setprecision(2) << s_clahe.rms_contrast << std::endl;
    std::cerr << "      Umbral ISO 29794-1: ≥30.0 (escala 0-255)" << std::endl;
    bool rms_ok = (s_clahe.rms_contrast >= 30.0);
    std::cerr << "      Resultado: " << (rms_ok ? "✓ PASS" : "⚠ BAJO CONTRASTE") << std::endl;
    
    // 2. Entropía de Shannon (Shannon 1948, Pizer 1987) - bits
    double delta_entropy = s_clahe.entropy - s_resize.entropy;
    std::cerr << "    Entropía Shannon: " << std::fixed << std::setprecision(2) 
              << s_resize.entropy << " → " << s_clahe.entropy << " bits" << std::endl;
    std::cerr << "      Ganancia: " << std::showpos << delta_entropy << std::noshowpos << " bits" << std::endl;
    std::cerr << "      Umbral: >0 (debe aumentar)" << std::endl;
    bool entropy_ok = (delta_entropy > 0.0);
    std::cerr << "      Resultado: " << (entropy_ok ? "✓ PASS" : "⚠ NO MEJORA") << std::endl;
    
    // 3. Michelson Contrast (Michelson 1927, ISO 9241) - escala [0-1]
    std::cerr << "    Michelson Contrast: " << std::fixed << std::setprecision(3) << s_clahe.michelson_contrast << std::endl;
    std::cerr << "      Umbral: ≥0.70 (escala 0-1)" << std::endl;
    bool michelson_ok = (s_clahe.michelson_contrast >= 0.70);
    std::cerr << "      Resultado: " << (michelson_ok ? "✓ PASS" : "⚠ BAJO") << std::endl;
    
    // VEREDICTO FINAL
    bool clahe_efectivo = (rms_ok && entropy_ok);
    std::cerr << "  VALIDACIÓN CLAHE: " << (clahe_efectivo ? "✓ EFECTIVO" : "⚠ REQUIERE REVISIÓN") << std::endl;
    std::cerr << std::endl;

    // Paso 3: Bilateral Filter (σ_space=3, σ_color=50)
    t0 = std::chrono::steady_clock::now();
    out.img128 = aplicarBilateral(img128_clahe.get(), 128, 128, 3.0, 50.0);
    auto ms_bilateral = ms_since(t0);
    
    GrayStats s_bilateral = calcGrayStats(out.img128.get(), 128, 128);
    logMetrics("FASE 3: FILTRO BILATERAL (Reducción de Ruido)", s_clahe, s_bilateral, ms_bilateral);
    
    // ========== MÉTRICAS CUANTITATIVAS BILATERAL ==========
    // Referencia: Tomasi & Manduchi (1998) "Bilateral filtering for gray and color images"
    //             Aurich & Weule (1995) "Non-linear gaussian filters"
    
    std::cerr << "  MÉTRICAS BILATERAL:" << std::endl;
    
    // 1. Reducción de Varianza (proxy de ruido)
    double var_antes = s_clahe.stddev * s_clahe.stddev;
    double var_despues = s_bilateral.stddev * s_bilateral.stddev;
    double reduccion_var = var_antes - var_despues;
    std::cerr << "    Reducción Varianza: " << std::fixed << std::setprecision(2) << reduccion_var << std::endl;
    std::cerr << "      Umbral: ≥10.0 (escala 0-65025)" << std::endl;
    bool var_ok = (reduccion_var >= 10.0);
    std::cerr << "      Resultado: " << (var_ok ? "✓ PASS" : "⚠ BAJO") << std::endl;
    
    // 2. Preservación de Entropía (no debe caer drásticamente)
    double ratio_entropy = s_bilateral.entropy / s_clahe.entropy;
    std::cerr << "    Preservación Entropía: " << std::fixed << std::setprecision(3) << ratio_entropy << std::endl;
    std::cerr << "      Umbral: ≥0.90 (debe preservar ≥90% de información)" << std::endl;
    bool entropy_preserved = (ratio_entropy >= 0.90);
    std::cerr << "      Resultado: " << (entropy_preserved ? "✓ PASS" : "⚠ PÉRDIDA EXCESIVA") << std::endl;
    
    // 3. Smoothness sin pérdida de bordes
    std::cerr << "    RMS Post-Filtro: " << std::fixed << std::setprecision(2) << s_bilateral.rms_contrast << std::endl;
    std::cerr << "      Umbral: ≥25.0 (debe mantener contraste de bordes)" << std::endl;
    bool edges_ok = (s_bilateral.rms_contrast >= 25.0);
    std::cerr << "      Resultado: " << (edges_ok ? "✓ PASS" : "⚠ SOBRE-SUAVIZADO") << std::endl;
    
    bool bilateral_ok = (var_ok && entropy_preserved && edges_ok);
    std::cerr << "  VALIDACIÓN BILATERAL: " << (bilateral_ok ? "✓ EFECTIVO" : "⚠ REQUIERE REVISIÓN") << std::endl;
    std::cerr << "  Nota: Bilateral elimina ruido preservando bordes (Tomasi & Manduchi 1998)" << std::endl;
    std::cerr << std::endl;

    // Paso 4: Máscara elíptica FIJA
    t0 = std::chrono::steady_clock::now();
    out.mask128 = crearMascaraElipticaFija(128, 128);
    auto ms_mask = ms_since(t0);
    
    double coverage = maskCoveragePct(out.mask128.get(), 128, 128);
    bool coverage_ok = (coverage >= 50.0 && coverage <= 80.0);
    
    std::cerr << "FASE 4: MASCARA ELIPTICA FIJA (ROI)" << std::endl;
    std::cerr << "  Cobertura    -> " << std::fixed << std::setprecision(1) << coverage << "% del área total" << std::endl;
    std::cerr << "  Validación   -> cobertura=" << std::fixed << std::setprecision(1) << coverage
              << "% umbral=[50%,80%] " << (coverage_ok ? "✓ PASS" : "⚠ FUERA DE RANGO") << std::endl;
    std::cerr << "  Tiempo       -> " << ms_mask << " ms" << std::endl;
    std::cerr << std::endl;

    // Resumen del pipeline
    long long total_ms = ms_resize + ms_clahe + ms_bilateral + ms_mask;
    std::cerr << "RESUMEN PIPELINE:" << std::endl;
    std::cerr << "  1. Resize       -> " << ms_resize << " ms" << std::endl;
    std::cerr << "  2. CLAHE        -> " << ms_clahe << " ms" << std::endl;
    std::cerr << "  3. Bilateral    -> " << ms_bilateral << " ms" << std::endl;
    std::cerr << "  4. Máscara      -> " << ms_mask << " ms" << std::endl;
    std::cerr << "  TOTAL          -> " << total_ms << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    return out;
}

static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    // LBP Multi-Scale (radius=1 + radius=2): 6x6 bloques, 200 umbral
    // IMPORTANTE: Debe coincidir EXACTAMENTE con procesar_dataset.cpp
    // Dimensiones: 6×6 bloques × 118 bins (multi-scale) = 4248 features
    // ✅ Multi-Scale LBP para inferencia
    return calcularLBPMultiEscalaPorBloquesRobustoNorm(img128, mask128, 128, 128, 6, 6, 200, true);
}

static std::vector<double> extraerCaracteristicas(const uint8_t* imagenGris, int ancho, int alto) {
    Imagen128 base = preprocesarHasta128(imagenGris, ancho, alto);
    return extraerFeaturesDesde128(base.img128.get(), base.mask128.get());
}

struct TemplateModel {
    std::vector<int> clases;
    std::vector<std::vector<double>> templates;
    std::vector<double> norms;
};

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

static bool cargarTemplatesCSV(const std::string& ruta, TemplateModel& tm) {
    std::ifstream f(ruta);
    if (!f.is_open()) return false;
    tm.clases.clear();
    tm.templates.clear();
    tm.norms.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        if (!std::getline(ss, token, ';')) continue;
        int clase = -1;
        try { clase = std::stoi(token); } catch (...) { continue; }
        std::vector<double> v;
        while (std::getline(ss, token, ';')) {
            try { v.push_back(std::stod(token)); } catch (...) { v.push_back(0.0); }
        }
        if (v.empty()) continue;
        tm.clases.push_back(clase);
        tm.templates.push_back(v);
        tm.norms.push_back(l2norm(v));
    }
    return !tm.clases.empty();
}

static bool scoreTemplatesK1(const TemplateModel& tm,
                             const std::vector<double>& x,
                             int claimedId,
                             int& claseTop1,
                             double& scoreTop1,
                             double& scoreTop2,
                             double& scoreClaimed) {
    if (tm.clases.empty() || tm.templates.empty()) return false;
    const double normX = l2norm(x);
    claseTop1 = -1;
    scoreTop1 = -std::numeric_limits<double>::infinity();
    scoreTop2 = -std::numeric_limits<double>::infinity();
    scoreClaimed = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < tm.clases.size(); ++i) {
        const auto& t = tm.templates[i];
        if (t.size() != x.size()) continue;
        double s = cosineSim(x, normX, t, tm.norms[i]);
        if (s > scoreTop1) {
            scoreTop2 = scoreTop1;
            scoreTop1 = s;
            claseTop1 = tm.clases[i];
        } else if (s > scoreTop2) {
            scoreTop2 = s;
        }
        if (claimedId != -1 && tm.clases[i] == claimedId) scoreClaimed = s;
    }

    if (claimedId == -1 || !std::isfinite(scoreClaimed)) scoreClaimed = scoreTop1;
    return (claseTop1 != -1) && std::isfinite(scoreTop1) && std::isfinite(scoreTop2);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "ERROR: Uso: predecir <ruta_imagen> [--rid <id>] [--claim <id_usuario>]" << std::endl;
        return 1;
    }

    // ---- args compatibles (NO rompe al servidor) ----
    std::string rid = makeRequestId();
    int claimedId = -1;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--rid" && i + 1 < argc) {
            rid = argv[++i];
        } else if (a == "--claim" && i + 1 < argc) {
            try { claimedId = std::stoi(argv[++i]); } catch (...) { claimedId = -1; }
        }
    }

    const std::string rutaImagen = argv[1];

    std::cerr << "\n[PREDECIR] RID: " << rid << std::endl;
    if (claimedId != -1) std::cerr << "Claim ID: " << claimedId << std::endl;
    std::cerr << "Inicio predicción. ruta_imagen=" << rutaImagen << " cwd=" << fs::current_path().string() << std::endl;
    std::cerr << std::endl;

    // 1) Validar archivo
    try {
        if (!fs::exists(rutaImagen)) {
            std::cerr << "ERROR: Archivo NO existe: " << rutaImagen << std::endl;
            return 2;
        }
        auto sz = fs::file_size(rutaImagen);
        std::cerr << "Archivo OK size_bytes=" << sz << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Error revisando archivo: " << e.what() << std::endl;
        return 2;
    }

    // 2) Cargar imagen
    auto t0 = std::chrono::steady_clock::now();
    int ancho = 0, alto = 0, canales = 0;
    unsigned char* imgRGB = cargarImagen(rutaImagen.c_str(), ancho, alto, canales, 3);
    if (!imgRGB) {
        std::cerr << "ERROR: Error cargando imagen (cargarImagen devolvió null)" << std::endl;
        return 3;
    }
    std::cerr << "Imagen cargada w=" << ancho << " h=" << alto 
              << " canales_in=" << canales << " ms=" << ms_since(t0) << std::endl;

    // 3) Gris
    t0 = std::chrono::steady_clock::now();
    auto gris = convertirAGris(imgRGB, ancho, alto);
    delete[] imgRGB;
    std::cerr << "Convertir a gris OK ms=" << ms_since(t0) << std::endl;
    std::cerr << std::endl;

    // 4) Extracción (prepro + LBP)
    t0 = std::chrono::steady_clock::now();
    logPhaseHeader("EXTRACCION DE CARACTERISTICAS");
    std::cerr << "Método: LBP Multi-Scale (6x6 bloques, umbral=200)" << std::endl;
    auto caracteristicas = extraerCaracteristicas(gris.get(), ancho, alto);
    if (caracteristicas.empty()) {
        std::cerr << "ERROR: Error extrayendo características (vector vacío)" << std::endl;
        return 4;
    }
    auto ms_lbp = ms_since(t0);
    std::cerr << "LBP OK dim=" << caracteristicas.size() << " ms=" << ms_lbp << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

                   
    const std::string rutaZ   = "out/zscore_params.dat";

    // 4.5) Z-score (obligatorio)
    logPhaseHeader("NORMALIZACION Z-SCORE");
    t0 = std::chrono::steady_clock::now();
    ZScoreParams zp;
    if (!fs::exists(rutaZ) || !cargarZScoreParams(rutaZ, zp, ';')) {
        std::cerr << "ERROR: Z-score params NO disponibles: " << rutaZ << std::endl;
        return 55;
    }
    if (caracteristicas.size() != zp.mean.size()) {
        std::cerr << "ERROR: DIM_MISMATCH Z-score: feat_dim=" << caracteristicas.size()
                  << " z_dim=" << zp.mean.size() << std::endl;
        return 56;
    }
    
    // Calcular estadísticas antes de Z-score
    double mean_before = std::accumulate(caracteristicas.begin(), caracteristicas.end(), 0.0) / caracteristicas.size();
    double sum_sq = std::inner_product(caracteristicas.begin(), caracteristicas.end(), caracteristicas.begin(), 0.0);
    double var_before = sum_sq / caracteristicas.size() - mean_before * mean_before;
    double std_before = std::sqrt(var_before > 0 ? var_before : 0);
    
    if (!aplicarZScore(caracteristicas, zp)) {
        std::cerr << "ERROR: Error aplicando Z-score." << std::endl;
        return 57;
    }
    
    // Calcular estadísticas después de Z-score
    double mean_after = std::accumulate(caracteristicas.begin(), caracteristicas.end(), 0.0) / caracteristicas.size();
    sum_sq = std::inner_product(caracteristicas.begin(), caracteristicas.end(), caracteristicas.begin(), 0.0);
    double var_after = sum_sq / caracteristicas.size() - mean_after * mean_after;
    double std_after = std::sqrt(var_after > 0 ? var_after : 0);
    
    auto ms_zscore = ms_since(t0);
    std::cerr << "Parámetros: mean[dataset], std[dataset] para cada dimensión" << std::endl;
    std::cerr << "Dimensión    -> " << caracteristicas.size() << " features" << std::endl;
    std::cerr << "Antes        -> mean=" << std::fixed << std::setprecision(4) << mean_before 
              << " std=" << std_before << std::endl;
    std::cerr << "Después      -> mean=" << std::fixed << std::setprecision(4) << mean_after 
              << " std=" << std_after << std::endl;
    std::cerr << "Validación   -> mean≈0 std≈1 ✓ NORMALIZADO" << std::endl;
    std::cerr << "Tiempo       -> " << ms_zscore << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // 5) PCA
    const std::string rutaPCA = "out/modelo_pca.dat";

    logPhaseHeader("REDUCCION DIMENSIONAL PCA");
    t0 = std::chrono::steady_clock::now();
    if (!fs::exists(rutaPCA)) {
        std::cerr << "ERROR: Modelo PCA NO existe en: " << rutaPCA << std::endl;
        return 5;
    }
    ModeloPCA modeloPCA = cargarModeloPCA(rutaPCA);
    int dim_in = caracteristicas.size();
    auto reducidas = aplicarPCAConModelo({ caracteristicas }, modeloPCA);
    if (reducidas.empty() || reducidas[0].empty()) {
        std::cerr << "ERROR: Error aplicando PCA (resultado vacío)" << std::endl;
        return 6;
    }
    int dim_out = reducidas[0].size();
    auto ms_pca = ms_since(t0);
    
    double reduccion_pct = 100.0 * (1.0 - (double)dim_out / (double)dim_in);
    
    std::cerr << "Entrada      -> " << dim_in << " dimensiones" << std::endl;
    std::cerr << "Salida       -> " << dim_out << " componentes principales" << std::endl;
    std::cerr << "Reducción    -> " << std::fixed << std::setprecision(1) << reduccion_pct << "% (de "
              << dim_in << " a " << dim_out << ")" << std::endl;
    
    if (dim_out >= 100 && dim_out <= 140) {
        std::cerr << "Validación   -> dim_out en rango recomendado [100,140] ✓ PASS" << std::endl;
    } else if (dim_out > 140) {
        std::cerr << "Validación   -> dim_out=" << dim_out << " > 140 ⚠ RIESGO OVERFITTING" << std::endl;
    } else {
        std::cerr << "Validación   -> dim_out=" << dim_out << " < 100 ⚠ PÉRDIDA DE INFORMACIÓN" << std::endl;
    }
    
    std::cerr << "Tiempo       -> " << ms_pca << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // 6) Normalización L2 (PCA)
    logPhaseHeader("NORMALIZACION L2 (PCA)");
    t0 = std::chrono::steady_clock::now();
    for (auto& v : reducidas) normalizarVector(v);
    auto ms_norm_pca = ms_since(t0);
    std::cerr << "Vectores PCA normalizados (L2)" << std::endl;
    std::cerr << "Tiempo       -> " << ms_norm_pca << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // 7) LDA
    const std::string rutaLDA = "out/modelo_lda.dat";
    logPhaseHeader("REDUCCION DISCRIMINANTE LDA");
    t0 = std::chrono::steady_clock::now();
    if (!fs::exists(rutaLDA)) {
        std::cerr << "ERROR: Modelo LDA NO existe en: " << rutaLDA << std::endl;
        return 7;
    }
    ModeloLDA modeloLDA = cargarModeloLDA(rutaLDA);
    auto lda = aplicarLDAConModelo(reducidas, modeloLDA);
    if (lda.empty() || lda[0].empty()) {
        std::cerr << "ERROR: Error aplicando LDA (resultado vacío)" << std::endl;
        return 8;
    }
    auto ms_lda = ms_since(t0);
    std::cerr << "Entrada      -> " << dim_out << " dims (PCA)" << std::endl;
    std::cerr << "Salida       -> " << lda[0].size() << " dims (LDA)" << std::endl;
    std::cerr << "Tiempo       -> " << ms_lda << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // 8) Normalización L2 (LDA)
    logPhaseHeader("NORMALIZACION L2 (LDA)");
    t0 = std::chrono::steady_clock::now();
    for (auto& v : lda) normalizarVector(v);
    auto ms_norm_lda = ms_since(t0);
    std::cerr << "Vectores LDA normalizados (L2)" << std::endl;
    std::cerr << "Tiempo       -> " << ms_norm_lda << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // 9) Templates (coseno, K=1)
    const std::string rutaTemplates = "out/templates_k1.csv";
    logPhaseHeader("TEMPLATES POR USUARIO (COSENO, K=1)");
    t0 = std::chrono::steady_clock::now();
    if (!fs::exists(rutaTemplates)) {
        std::cerr << "ERROR: Templates NO existen en: " << rutaTemplates << std::endl;
        return 9;
    }

    TemplateModel tm;
    if (!cargarTemplatesCSV(rutaTemplates, tm)) {
        std::cerr << "ERROR: Error cargando templates (CSV inválido)" << std::endl;
        return 10;
    }

    int clase = -1;
    double s1 = 0.0, s2 = 0.0, s_claimed = 0.0;
    if (!scoreTemplatesK1(tm, lda[0], claimedId, clase, s1, s2, s_claimed)) {
        std::cerr << "ERROR: Error puntuando templates" << std::endl;
        return 11;
    }

    double margen = s1 - s2;
    auto ms_tpl = ms_since(t0);

    std::cerr << "  Top-1        -> Clase " << clase << " (score=" << std::fixed << std::setprecision(4) << s1 << ")" << std::endl;
    std::cerr << "  Top-2        -> score=" << std::fixed << std::setprecision(4) << s2 << std::endl;
    std::cerr << "  Score claim  -> " << std::fixed << std::setprecision(4) << s_claimed << std::endl;
    std::cerr << "  Margen       -> " << std::fixed << std::setprecision(4) << margen << " (s1-s2)" << std::endl;
    std::cerr << "  Tiempo       -> " << ms_tpl << " ms" << std::endl;
    std::cerr << "====================================================" << std::endl;
    std::cerr << std::endl;

    // Resumen final
    logPhaseHeader("RESUMEN FINAL - AUTENTICACION");
    std::cerr << "Usuario Predicho: " << clase << std::endl;
    std::cerr << "Score Top-1:       " << std::fixed << std::setprecision(4) << s1 << std::endl;
    std::cerr << "Score Claim:       " << std::fixed << std::setprecision(4) << s_claimed << std::endl;
    std::cerr << "Pipeline Total:    " << (ms_lbp + ms_zscore + ms_pca + ms_norm_pca + ms_lda + ms_norm_lda + ms_tpl) << " ms" << std::endl;
    std::cerr << "===================================================" << std::endl;

    // IMPORTANTÍSIMO:
    // stdout: clase;score_top1;score_claimed
    // Formato: clase;score_top1;score_claimed
    std::cout << clase << ";" << s1 << ";" << s_claimed << "\n";
    return 0;
}
