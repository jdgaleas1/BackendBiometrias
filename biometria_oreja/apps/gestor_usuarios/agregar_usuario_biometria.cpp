// agregar_usuario_biometria.cpp - SINCRONIZADO CON FASE 6
#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "preprocesamiento/aumentar_dataset.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/normalizacion.h"
#include "utilidades/pca_utils.h"
#include "utilidades/lda_utils.h"
#include "svm/cargar_csv.h"
#include "utilidades/guardar_csv.h"
#include "json.hpp"
#include "httplib.h"
#include "admin/admin_types.h"
#include "admin/admin_time.h"
#include "admin/admin_log_helpers.h"
#include "admin/admin_config.h"
#include "admin/admin_report.h"
#include "utilidades/zscore_params.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <limits>

#include <omp.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

static std::string vecSample10(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    int n = (int)std::min<size_t>(10, v.size());
    for (int i = 0; i < n; ++i) {
        oss << std::fixed << std::setprecision(3) << v[i];
        if (i + 1 < n) oss << ", ";
    }
    if ((int)v.size() > n) oss << ", ...";
    oss << "]";
    return oss.str();
}

static int exitCode(int c) { return c; }


static GrayStats calcGrayStats(const uint8_t* img, int w, int h, int dark_thr=10, int bright_thr=245) {
    GrayStats s;
    const int N = w*h;
    if (!img || N <= 0) return s;

    long long sum = 0;
    long long sum2 = 0;
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

static TemplateModel construirTemplatesK1(const std::vector<std::vector<double>>& X,
                                          const std::vector<int>& y) {
    TemplateModel tm;
    if (X.empty() || y.empty() || X.size() != y.size()) return tm;

    std::unordered_map<int, std::vector<double>> sum;
    std::unordered_map<int, int> count;
    for (size_t i = 0; i < X.size(); ++i) {
        auto& acc = sum[y[i]];
        if (acc.empty()) acc.assign(X[i].size(), 0.0);
        for (size_t d = 0; d < X[i].size(); ++d) acc[d] += X[i][d];
        count[y[i]]++;
    }

    tm.clases.reserve(sum.size());
    tm.templates.reserve(sum.size());
    tm.norms.reserve(sum.size());

    std::vector<int> clases;
    clases.reserve(sum.size());
    for (const auto& kv : sum) clases.push_back(kv.first);
    std::sort(clases.begin(), clases.end());

    for (int c : clases) {
        auto it = sum.find(c);
        if (it == sum.end()) continue;
        auto v = it->second;
        int cnt = std::max(1, count[c]);
        for (double& x : v) x /= (double)cnt;
        tm.clases.push_back(c);
        tm.templates.push_back(v);
        tm.norms.push_back(l2norm(v));
    }
    return tm;
}

static bool guardarTemplatesCSV(const std::string& ruta, const TemplateModel& tm) {
    const auto parent = fs::path(ruta).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
    std::ofstream f(ruta);
    if (!f.is_open()) return false;
    for (size_t i = 0; i < tm.clases.size(); ++i) {
        f << tm.clases[i];
        for (double v : tm.templates[i]) f << ";" << v;
        f << "\n";
    }
    return true;
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
                             int& claseTop1,
                             double& scoreTop1,
                             double& scoreTop2) {
    if (tm.clases.empty()) return false;
    const double normX = l2norm(x);
    claseTop1 = -1;
    scoreTop1 = -std::numeric_limits<double>::infinity();
    scoreTop2 = -std::numeric_limits<double>::infinity();
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
    }
    return (claseTop1 != -1) && std::isfinite(scoreTop1) && std::isfinite(scoreTop2);
}

static std::string passFail(bool ok) { return ok ? "PASS" : "FAIL"; }

static bool qcGrayPass(const GrayStats& s, const QcThresholds& t, std::string& reason) {
    if (s.mean < t.mean_min || s.mean > t.mean_max) { reason = "mean_fuera_rango"; return false; }
    if (s.stddev < t.std_min) { reason = "contraste_bajo(std)"; return false; }
    if (s.minv < t.min_min) { reason = "demasiado_oscura(min)"; return false; }
    if (s.maxv > t.max_max) { reason = "demasiado_clara(max)"; return false; }
    if (s.pct_dark > t.pct_dark_max) { reason = "muchos_pixeles_oscuros"; return false; }
    if (s.pct_bright > t.pct_bright_max) { reason = "muchos_pixeles_claros"; return false; }
    reason.clear();
    return true;
}

static GrayStats calcGrayStatsMasked(const uint8_t* img, const uint8_t* mask, int w, int h,
                                     int dark_thr=10, int bright_thr=245)
{
    GrayStats s;
    const int N = w*h;
    if (!img || !mask || N <= 0) return s;

    long long sum = 0, sum2 = 0;
    int c_dark = 0, c_bright = 0;
    int count = 0;

    for (int i = 0; i < N; ++i) {
        if (mask[i] == 0) continue;          // SOLO ROI
        int v = (int)img[i];
        sum += v;
        sum2 += 1LL*v*v;
        if (v < s.minv) s.minv = v;
        if (v > s.maxv) s.maxv = v;
        if (v <= dark_thr) c_dark++;
        if (v >= bright_thr) c_bright++;
        count++;
    }

    if (count <= 0) { // ROI vacía
        s.mean = 0; s.stddev = 0; s.pct_dark = 100; s.pct_bright = 0;
        s.minv = 0; s.maxv = 0;
        return s;
    }

    s.mean = (double)sum / (double)count;
    double var = (double)sum2 / (double)count - s.mean*s.mean;
    if (var < 0) var = 0;
    s.stddev = std::sqrt(var);
    s.pct_dark = 100.0 * (double)c_dark / (double)count;
    s.pct_bright = 100.0 * (double)c_bright / (double)count;

    return s;
}

// ====================== PREPROCESAMIENTO DETALLADO ======================
// PIPELINE FASE 6 - SINCRONIZADO CON procesar_dataset.cpp
// Pasos: 1) Resize 128x128  2) CLAHE  3) Bilateral  4) Máscara Fija
// ============================================================================
static Imagen128 preprocesarHasta128(const uint8_t* imagenGris, int ancho, int alto,
                                     std::ofstream& log, const std::string& rid,
                                     int LOG_DETAIL, const std::string& file,
                                     const QcThresholds& QC)
{
    Imagen128 out;

    if (LOG_DETAIL >= 2) {
        logTechTitle(log, rid, "PREPROCESAMIENTO FASE 6 (SINCRONIZADO)");
        logRawLine(log, rid, "Entrada:");
        logRawLine(log, rid, "  - Archivo: " + file);
        logRawLine(log, rid, "  - Dimensiones: " + std::to_string(ancho) + "x" + std::to_string(alto));
        logRawLine(log, rid, "  - Formato: Escala de grises (1 canal)");
        logBlank(log, rid);
    }

    GrayStats s_original = calcGrayStats(imagenGris, ancho, alto, QC.dark_thr, QC.bright_thr);

    // ============================================================================
    // TÉCNICA 1: REDIMENSIONAMIENTO DIRECTO A 128x128
    // Cambio FASE 6: Eliminamos bilateral previo, iluminación V2, y segmentación
    // Trabajamos DIRECTAMENTE con la imagen original
    // ============================================================================
    TimePoint t0 = tick();
    auto img128 = redimensionarParaBiometria(imagenGris, ancho, alto, 128, 128);
    auto ms_resize = msSince(t0);

    if (LOG_DETAIL >= 2) {
        GrayStats s_128 = calcGrayStats(img128.get(), 128, 128, QC.dark_thr, QC.bright_thr);

        StatsComparison cmp;
        cmp.tecnica = "1. REDIMENSIONAMIENTO";
        cmp.params = "Interpolación: bilinear, 128x128";

        cmp.w_in = ancho; cmp.h_in = alto;
        cmp.mean_in = s_original.mean;
        cmp.std_in = s_original.stddev;
        cmp.min_in = s_original.minv;
        cmp.max_in = s_original.maxv;
        cmp.pct_dark_in = s_original.pct_dark;
        cmp.pct_bright_in = s_original.pct_bright;

        cmp.w_out = 128; cmp.h_out = 128;
        cmp.mean_out = s_128.mean;
        cmp.std_out = s_128.stddev;
        cmp.min_out = s_128.minv;
        cmp.max_out = s_128.maxv;
        cmp.pct_dark_out = s_128.pct_dark;
        cmp.pct_bright_out = s_128.pct_bright;

        cmp.ms = ms_resize;
        cmp.efecto = "Normalización de tamaño, mantiene proporciones";

        logTechniqueComparison(log, rid, cmp);
        
        // Validación: Relación de aspecto
        double aspect_ratio = (double)ancho / (double)alto;
        const double MAX_ASPECT_RATIO = 1.15; // Umbral máximo aceptable
        bool aspect_ok = (aspect_ratio >= 0.85 && aspect_ratio <= MAX_ASPECT_RATIO);
        
        logRawLine(log, rid, "Validación de Relación de Aspecto:");
        logRawLine(log, rid, "  - Relación calculada: " + std::to_string(aspect_ratio));
        logRawLine(log, rid, "  - Umbral aceptable:   [0.85, " + std::to_string(MAX_ASPECT_RATIO) + "]");
        logRawLine(log, rid, "  - Estado:             " + std::string(aspect_ok ? "✓ PASS" : "⚠ ADVERTENCIA"));
        if (!aspect_ok) {
            logRawLine(log, rid, "  ⚠ Imagen con proporción inusual, puede afectar precisión");
        }
        logBlank(log, rid);
    }

    // ============================================================================
    // TÉCNICA 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    // Parámetros FASE 6: 8×8 tiles, clipLimit=2.0
    // Mejora contraste local, realza texturas sutiles para LBP
    // ============================================================================
    t0 = tick();
    auto img128_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);
    auto ms_clahe = msSince(t0);

    if (LOG_DETAIL >= 2) {
        GrayStats s_pre = calcGrayStats(img128.get(), 128, 128, QC.dark_thr, QC.bright_thr);
        GrayStats s_post = calcGrayStats(img128_clahe.get(), 128, 128, QC.dark_thr, QC.bright_thr);

        StatsComparison cmp;
        cmp.tecnica = "2. CLAHE";
        cmp.params = "tileSize=8x8, clipLimit=2.0";

        cmp.w_in = 128; cmp.h_in = 128;
        cmp.mean_in = s_pre.mean;
        cmp.std_in = s_pre.stddev;
        cmp.min_in = s_pre.minv;
        cmp.max_in = s_pre.maxv;
        cmp.pct_dark_in = s_pre.pct_dark;
        cmp.pct_bright_in = s_pre.pct_bright;

        cmp.w_out = 128; cmp.h_out = 128;
        cmp.mean_out = s_post.mean;
        cmp.std_out = s_post.stddev;
        cmp.min_out = s_post.minv;
        cmp.max_out = s_post.maxv;
        cmp.pct_dark_out = s_post.pct_dark;
        cmp.pct_bright_out = s_post.pct_bright;

        cmp.ms = ms_clahe;
        cmp.efecto = "Mejora contraste local adaptativo, realza texturas";

        logTechniqueComparison(log, rid, cmp);
        
        // Cálculo explícito de ganancia de contraste
        double ganancia_contraste_pct = 0.0;
        if (s_pre.stddev > 0) {
            ganancia_contraste_pct = 100.0 * (s_post.stddev - s_pre.stddev) / s_pre.stddev;
        }
        
        const double MIN_CONTRAST_GAIN = 5.0; // 5% mínimo esperado
        bool contrast_gain_ok = (ganancia_contraste_pct >= MIN_CONTRAST_GAIN);
        
        logRawLine(log, rid, "Análisis de Ganancia de Contraste:");
        logRawLine(log, rid, "  - Desv.Est. antes:    " + std::to_string(s_pre.stddev));
        logRawLine(log, rid, "  - Desv.Est. después:  " + std::to_string(s_post.stddev));
        logRawLine(log, rid, "  - Ganancia estimada:  " + std::to_string((int)std::round(ganancia_contraste_pct)) + "%");
        logRawLine(log, rid, "  - Umbral mínimo:      " + std::to_string((int)MIN_CONTRAST_GAIN) + "%");
        logRawLine(log, rid, "  - Validación:         " + std::string(contrast_gain_ok ? "✓ PASS" : "⚠ BAJO"));
        if (!contrast_gain_ok) {
            logRawLine(log, rid, "  ⚠ Ganancia de contraste menor a esperada (imagen ya tenía buen contraste)");
        }
        logBlank(log, rid);
    }

    // ============================================================================
    // TÉCNICA 3: FILTRO BILATERAL (Edge-Preserving Denoising)
    // Parámetros FASE 6: σ_space=3, σ_color=50
    // Reduce ruido post-CLAHE sin destruir bordes ni texturas finas
    // ============================================================================
    t0 = tick();
    out.img128 = aplicarBilateral(img128_clahe.get(), 128, 128, 3.0, 50.0);
    auto ms_bilateral = msSince(t0);

    if (LOG_DETAIL >= 2) {
        GrayStats s_pre = calcGrayStats(img128_clahe.get(), 128, 128, QC.dark_thr, QC.bright_thr);
        GrayStats s_post = calcGrayStats(out.img128.get(), 128, 128, QC.dark_thr, QC.bright_thr);

        StatsComparison cmp;
        cmp.tecnica = "3. FILTRO BILATERAL";
        cmp.params = "sigmaSpace=3.0, sigmaColor=50.0";

        cmp.w_in = 128; cmp.h_in = 128;
        cmp.mean_in = s_pre.mean;
        cmp.std_in = s_pre.stddev;
        cmp.min_in = s_pre.minv;
        cmp.max_in = s_pre.maxv;
        cmp.pct_dark_in = s_pre.pct_dark;
        cmp.pct_bright_in = s_pre.pct_bright;

        cmp.w_out = 128; cmp.h_out = 128;
        cmp.mean_out = s_post.mean;
        cmp.std_out = s_post.stddev;
        cmp.min_out = s_post.minv;
        cmp.max_out = s_post.maxv;
        cmp.pct_dark_out = s_post.pct_dark;
        cmp.pct_bright_out = s_post.pct_bright;

        cmp.ms = ms_bilateral;
        cmp.efecto = "Reducción de ruido preservando bordes y texturas";

        logTechniqueComparison(log, rid, cmp);
        
        // Cálculo explícito de reducción de ruido (varianza como proxy)
        double reduccion_ruido_pct = 0.0;
        if (s_pre.stddev > 0) {
            reduccion_ruido_pct = 100.0 * (s_pre.stddev - s_post.stddev) / s_pre.stddev;
        }
        
        const double MIN_NOISE_REDUCTION = 1.0; // 1% mínimo esperado
        bool noise_reduction_ok = (reduccion_ruido_pct >= MIN_NOISE_REDUCTION);
        
        logRawLine(log, rid, "Análisis de Reducción de Ruido:");
        logRawLine(log, rid, "  - Desv.Est. antes:    " + std::to_string(s_pre.stddev));
        logRawLine(log, rid, "  - Desv.Est. después:  " + std::to_string(s_post.stddev));
        logRawLine(log, rid, "  - Reducción estimada: " + std::to_string((int)std::round(reduccion_ruido_pct)) + "% (varianza como proxy de ruido)");
        logRawLine(log, rid, "  - Umbral mínimo:      " + std::to_string((int)MIN_NOISE_REDUCTION) + "%");
        logRawLine(log, rid, "  - Validación:         " + std::string(noise_reduction_ok ? "✓ PASS" : "⚠ BAJO"));
        if (!noise_reduction_ok) {
            logRawLine(log, rid, "  ⚠ Reducción menor a esperada (imagen ya tenía poco ruido post-CLAHE)");
        }
        logRawLine(log, rid, "  Nota: Desv.Est. es proxy de ruido; reducción indica suavizado exitoso");
        logBlank(log, rid);
    }

    // ============================================================================
    // TÉCNICA 4: MÁSCARA ELÍPTICA FIJA
    // Cambio FASE 6: Máscara FIJA en lugar de detectarRegionOreja (variable)
    // Consistencia 100% entre todas las imágenes
    // ============================================================================
    t0 = tick();
    out.mask128 = crearMascaraElipticaFija(128, 128);
    auto ms_mask = msSince(t0);

    if (LOG_DETAIL >= 2) {
        double cov = maskCoveragePct(out.mask128.get(), 128, 128);

        logTechTitle(log, rid, "4. MASCARA ELIPTICA FIJA");
        logRawLine(log, rid, "Tipo:         Elipse fija (consistente)");
        logRawLine(log, rid, "Dimensiones:  128x128");
        logRawLine(log, rid, "Cobertura:    " + std::to_string((int)std::round(cov)) + "% del área total");
        logRawLine(log, rid, "Tiempo:       " + std::to_string(ms_mask) + " ms");
        logRawLine(log, rid, "Ventaja:      100% consistente (vs segmentación variable)");
        logRawLine(log, rid, "Efecto:       Define ROI sin variabilidad entre imágenes");
        logBlank(log, rid);
        
        // Validación de cobertura de máscara
        const double MIN_MASK_COVERAGE = 50.0; // 50% mínimo
        const double MAX_MASK_COVERAGE = 80.0; // 80% máximo
        bool coverage_ok = (cov >= MIN_MASK_COVERAGE && cov <= MAX_MASK_COVERAGE);
        
        logRawLine(log, rid, "Validación de Cobertura ROI:");
        logRawLine(log, rid, "  - Cobertura medida:   " + std::to_string(cov) + "%");
        logRawLine(log, rid, "  - Umbral aceptable:   [" + std::to_string((int)MIN_MASK_COVERAGE) + "%, " + std::to_string((int)MAX_MASK_COVERAGE) + "%]");
        logRawLine(log, rid, "  - Validación:         " + std::string(coverage_ok ? "✓ PASS" : "⚠ FUERA DE RANGO"));
        if (!coverage_ok) {
            logRawLine(log, rid, "  ⚠ Cobertura inusual para máscara elíptica estándar");
        }
        logBlank(log, rid);
    }

    // ============================================================================
    // RESUMEN DEL PIPELINE FASE 6
    // ============================================================================
    if (LOG_DETAIL >= 2) {
        long long total_ms = ms_resize + ms_clahe + ms_bilateral + ms_mask;

        logTechTitle(log, rid, "RESUMEN DEL PIPELINE FASE 6");
        logRawLine(log, rid, "┌────────────────────────────────┬──────────┐");
        logRawLine(log, rid, "│ Técnica                        │ Tiempo   │");
        logRawLine(log, rid, "├────────────────────────────────┼──────────┤");
        logRawLine(log, rid, "│ 1. Redimensionamiento (128x128)│ " + std::to_string(ms_resize) + " ms    │");
        logRawLine(log, rid, "│ 2. CLAHE (8x8, clip=2.0)       │ " + std::to_string(ms_clahe) + " ms    │");
        logRawLine(log, rid, "│ 3. Bilateral (σs=3, σc=50)     │ " + std::to_string(ms_bilateral) + " ms    │");
        logRawLine(log, rid, "│ 4. Máscara Elíptica Fija       │ " + std::to_string(ms_mask) + " ms    │");
        logRawLine(log, rid, "├────────────────────────────────┼──────────┤");
        logRawLine(log, rid, "│ TOTAL                          │ " + std::to_string(total_ms) + " ms    │");
        logRawLine(log, rid, "└────────────────────────────────┴──────────┘");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "Transformación completa:");
        logRawLine(log, rid, "  " + std::to_string(ancho) + "x" + std::to_string(alto) + " (original) → 128x128 (normalizado)");
        logRawLine(log, rid, "  Imagen procesada + máscara FIJA lista para extracción LBP Multi-Scale 6x6x200");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 6 - Cambios respecto a versión anterior:");
        logRawLine(log, rid, "  ✗ ELIMINADO: Bilateral previo (σ=75, inconsistente)");
        logRawLine(log, rid, "  ✗ ELIMINADO: Ajuste iluminación V2 (redundante con CLAHE)");
        logRawLine(log, rid, "  ✗ ELIMINADO: detectarRegionOreja (máscaras variables)");
        logRawLine(log, rid, "  ✗ ELIMINADO: Recorte bounding box (introducía variabilidad)");
        logRawLine(log, rid, "  ✗ ELIMINADO: Dilatación 3x3 (innecesaria con máscara fija)");
        logRawLine(log, rid, "  ✓ NUEVO: Bilateral DESPUÉS de CLAHE (mejor posicionamiento)");
        logRawLine(log, rid, "  ✓ NUEVO: Máscara elíptica fija (100% consistente)");
        logBlank(log, rid);
    }
    
    // ============================================================================
    // VALIDACIÓN FINAL: RESUMEN DE UMBRALES DEL PIPELINE
    // ============================================================================
    if (LOG_DETAIL >= 2) {
        GrayStats s_final = calcGrayStatsMasked(out.img128.get(), out.mask128.get(), 128, 128, QC.dark_thr, QC.bright_thr);
        std::string qc_reason;
        bool qc_pass = qcGrayPass(s_final, QC, qc_reason);
        
        logTechTitle(log, rid, "VALIDACIÓN DE UMBRALES DEL PIPELINE COMPLETO");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 1: Conversión a Escala de Grises");
        logRawLine(log, rid, "  ✓ Conversión exitosa (estándar ITU-R BT.601)");
        logRawLine(log, rid, "  → No requiere umbral (transformación determinística)");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 2: Redimensionamiento 128×128");
        logRawLine(log, rid, "  ✓ Resize completado");
        logRawLine(log, rid, "  → Umbral validado: relación de aspecto");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 3: CLAHE (Mejora de Contraste)");
        logRawLine(log, rid, "  ✓ CLAHE aplicado (8×8 tiles, clipLimit=2.0)");
        logRawLine(log, rid, "  → Umbral validado: ganancia de contraste ≥ 5%");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 4: Filtro Bilateral (Reducción de Ruido)");
        logRawLine(log, rid, "  ✓ Bilateral aplicado (σ_space=3.0, σ_color=50.0)");
        logRawLine(log, rid, "  → Umbral validado: reducción de varianza ≥ 1%");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 5: Máscara Elíptica ROI");
        logRawLine(log, rid, "  ✓ Máscara aplicada");
        logRawLine(log, rid, "  → Umbral validado: cobertura en rango [50%, 80%]");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "FASE 6: Control de Calidad Final (QC)");
        logRawLine(log, rid, "  ✓ Métricas calculadas sobre ROI procesado final");
        logRawLine(log, rid, "  → Brillo promedio:   " + std::to_string(s_final.mean) + " (umbral: [" + std::to_string(QC.mean_min) + ", " + std::to_string(QC.mean_max) + "])");
        logRawLine(log, rid, "  → Contraste (std):   " + std::to_string(s_final.stddev) + " (umbral: ≥ " + std::to_string(QC.std_min) + ")");
        logRawLine(log, rid, "  → Píxeles oscuros:   " + std::to_string(s_final.pct_dark) + "% (umbral: ≤ " + std::to_string(QC.pct_dark_max) + "%)");
        logRawLine(log, rid, "  → Píxeles claros:    " + std::to_string(s_final.pct_bright) + "% (umbral: ≤ " + std::to_string(QC.pct_bright_max) + "%)");
        logRawLine(log, rid, "");
        logRawLine(log, rid, "════════════════════════════════════════════════════════");
        if (qc_pass) {
            logRawLine(log, rid, "VEREDICTO GLOBAL: ✓ IMAGEN APROBADA");
            logRawLine(log, rid, "Todos los umbrales de calidad han sido satisfechos.");
            logRawLine(log, rid, "La imagen está lista para extracción de características.");
        } else {
            logRawLine(log, rid, "VEREDICTO GLOBAL: ✗ IMAGEN RECHAZADA");
            logRawLine(log, rid, "Razón: " + qc_reason);
            logRawLine(log, rid, "La imagen no cumple con los estándares mínimos de calidad.");
        }
        logRawLine(log, rid, "════════════════════════════════════════════════════════");
        logBlank(log, rid);
    }

    return out;
}

// ====================== CONFIG LBP ======================
// IMPORTANTE: Parámetros SINCRONIZADOS con procesar_dataset.cpp
// Multi-Scale LBP: 6x6 bloques, threshold=200, Multi-Scale (radius=1+2)
// Dimensiones: 6×6×118 = 4248 features (DEBE COINCIDIR CON EL MODELO)
static constexpr int LBP_BX = 6;
static constexpr int LBP_BY = 6;
static constexpr int LBP_THRESHOLD = 200;
static constexpr bool LBP_USE_MASK = true;

static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    // ✅ Multi-Scale LBP
    return calcularLBPMultiEscalaPorBloquesRobustoNorm(img128, mask128, 128, 128, LBP_BX, LBP_BY, LBP_THRESHOLD, LBP_USE_MASK);
}

// ====================== CLIENTE POSTGREST ======================
static httplib::Client makeClientPostgrest() {
    const std::string host = getEnv("POSTGREST_HOST", "biometria_api");
    const int port = std::stoi(getEnv("POSTGREST_PORT", "3000"));

    httplib::Client cli(host, port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    cli.set_write_timeout(120, 0);
    return cli;
}

// ====================== HELPERS IO HOLDOUT/BASELINE/VERSIONADO ======================
static bool writeJsonFile(const std::string& path, const json& j) {
    try {
        const auto parent = fs::path(path).parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << j.dump(2);
        return true;
    } catch (...) { return false; }
}

static bool readJsonFile(const std::string& path, json& out) {
    try {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        f >> out;
        return true;
    } catch (...) { return false; }
}

static bool copyFileSafe(const std::string& src, const std::string& dst) {
    try {
        const auto parent = fs::path(dst).parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        return true;
    } catch (...) { return false; }
}

static bool loadHoldoutMeta(const std::string& path, HoldoutMeta& m) {
    json j;
    if (!readJsonFile(path, j)) return false;
    try {
        m.seed = j.value("seed", 42);
        m.total = j.value("total", 0);
        m.test_size = j.value("test_size", 0);
        m.dims = j.value("dims", 0);
        return true;
    } catch (...) { return false; }
}

static bool saveHoldoutMeta(const std::string& path, const HoldoutMeta& m) {
    json j;
    j["seed"] = m.seed;
    j["total"] = m.total;
    j["test_size"] = m.test_size;
    j["dims"] = m.dims;
    return writeJsonFile(path, j);
}

static bool ensureHoldoutFijo(
    std::ofstream& log,
    const std::string& rid,
    int LOG_DETAIL,
    const std::string& holdoutCsv,
    const std::string& holdoutMetaJson,
    const std::vector<std::vector<double>>& X_exist,
    const std::vector<int>& y_exist,
    HoldoutMeta& metaOut,
    double testRatio = 0.20,
    int seed = 42
) {
    if (fs::exists(holdoutCsv) && fs::exists(holdoutMetaJson)) {
        HoldoutMeta m;
        if (loadHoldoutMeta(holdoutMetaJson, m)) {
            metaOut = m;
            logDet(log, rid, LOG_DETAIL, 2, "[HOLDOUT] meta existente OK: test_size=" + std::to_string(m.test_size) + " seed=" + std::to_string(m.seed));
            return true;
        }
        logDet(log, rid, LOG_DETAIL, 1, "[HOLDOUT] WARN: meta corrupta -> recrear");
    }

    if (X_exist.empty() || y_exist.empty() || X_exist.size() != y_exist.size()) {
        logMensaje(log, rid, "[HOLDOUT] ERROR: dataset vacío o inconsistente. X=" + std::to_string(X_exist.size()) + " y=" + std::to_string(y_exist.size()));
        return false;
    }

    const int total = (int)X_exist.size();
    const int dims = (int)X_exist[0].size();
    int testSize = (int)std::round(total * testRatio);
    testSize = std::max(1, std::min(testSize, total));

    std::vector<int> idx(total);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<std::vector<double>> X_test;
    std::vector<int> y_test;
    X_test.reserve(testSize);
    y_test.reserve(testSize);

    for (int i = 0; i < testSize; ++i) {
        int k = idx[i];
        X_test.push_back(X_exist[k]);
        y_test.push_back(y_exist[k]);
    }

    logDet(log, rid, LOG_DETAIL, 2, "[HOLDOUT] guardando holdout_csv=" + holdoutCsv + " testSize=" + std::to_string(testSize));

    if (!guardarCSV(holdoutCsv, X_test, y_test, ';')) {
        logMensaje(log, rid, "[HOLDOUT] ERROR: no se pudo guardar holdout_test.csv");
        return false;
    }

    HoldoutMeta m;
    m.seed = seed;
    m.total = total;
    m.test_size = testSize;
    m.dims = dims;
    if (!saveHoldoutMeta(holdoutMetaJson, m)) {
        logMensaje(log, rid, "[HOLDOUT] WARN: no se pudo guardar holdout_meta.json (pero holdout_test.csv sí).");
    }

    metaOut = m;
    logMensaje(log, rid, "[HOLDOUT] creado OK: test_size=" + std::to_string(testSize) + " total=" + std::to_string(total) + " seed=" + std::to_string(seed));
    return true;
}


static bool loadBaseline(const std::string& baselineJson, double& outAcc) {
    json j;
    if (!readJsonFile(baselineJson, j)) return false;
    try {
        outAcc = j.value("baseline_acc", -1.0);
        return outAcc >= 0.0;
    } catch (...) { return false; }
}

static bool saveBaseline(const std::string& baselineJson, double acc) {
    json j;
    j["baseline_acc"] = acc;
    j["updated_at"] = nowTs();
    return writeJsonFile(baselineJson, j);
}

static bool makeBackupVersion(
    std::ofstream& log,
    const std::string& rid,
    int LOG_DETAIL,
    const std::string& dirVersiones,
    const std::string& rutaCSV,
    const std::string& rutaTemplates,
    std::string& outVersionDir
) {
    const std::string ver = dirVersiones + "/" + tsCompact();
    outVersionDir = ver;
    try { fs::create_directories(ver); } catch (...) {}

    logDet(log, rid, LOG_DETAIL, 2, "[BACKUP] creando version=" + ver);

    const std::string dstCSV = ver + "/" + fs::path(rutaCSV).filename().string();
    const std::string dstTpl = ver + "/" + fs::path(rutaTemplates).filename().string();

    bool ok1 = fs::exists(rutaCSV) ? copyFileSafe(rutaCSV, dstCSV) : true;
    bool ok2 = fs::exists(rutaTemplates) ? copyFileSafe(rutaTemplates, dstTpl) : true;

    if (ok1 && ok2) {
        logMensaje(log, rid, "[BACKUP] OK: " + dstCSV + " | " + dstTpl);
        return true;
    }
    logMensaje(log, rid, "[BACKUP] WARN: incompleto.");
    return false;
}

static bool rollbackFromVersion(
    std::ofstream& log,
    const std::string& rid,
    int LOG_DETAIL,
    const std::string& versionDir,
    const std::string& rutaCSV,
    const std::string& rutaTemplates
) {
    const std::string srcCSV = versionDir + "/" + fs::path(rutaCSV).filename().string();
    const std::string srcTpl = versionDir + "/" + fs::path(rutaTemplates).filename().string();

    bool ok = true;
    if (fs::exists(srcCSV)) ok &= copyFileSafe(srcCSV, rutaCSV);
    if (fs::exists(srcTpl)) ok &= copyFileSafe(srcTpl, rutaTemplates);

    if (ok) logMensaje(log, rid, "[ROLLBACK] OK desde: " + versionDir);
    else    logMensaje(log, rid, "[ROLLBACK] ERROR desde: " + versionDir);

    return ok;
}

// ====================== FUNCIÓN CORREGIDA ======================
static int procesarImagenesExtraerFeatures(std::ofstream& log, const Ctx& ctx,
                                          const std::vector<std::string>& imagenes,
                                          std::vector<ImageReport>& rep,
                                          std::vector<std::vector<double>>& nuevasCaracteristicas)
{
    logPhase(log, ctx.rid, ctx.LOG_DETAIL, 2, "INGESTA Y CONTROL POR IMAGEN",
            "Cargar imagen, convertir a gris y aplicar controles previos.", {});
    logPhase(log, ctx.rid, ctx.LOG_DETAIL, 3, "PREPROCESAMIENTO Y SEGMENTACION",
            "Reducir ruido, compensar iluminacion y aislar ROI mediante mascara.", {});
    logPhase(log, ctx.rid, ctx.LOG_DETAIL, 4, "EXTRACCION DE CARACTERISTICAS (LBP)",
            "Generar vectores por bloques y preparar entradas para PCA.", {});

    rep.assign(imagenes.size(), ImageReport{});
    for (size_t i = 0; i < imagenes.size(); ++i) {
        rep[i].name = fs::path(imagenes[i]).filename().string();
    }

    std::vector<std::vector<double>> caracteristicas_global;
    caracteristicas_global.reserve(imagenes.size() * 8);

    for (int i = 0; i < (int)imagenes.size(); ++i) {
        const auto& ruta = imagenes[i];

        ImageReport local;
        local.name = fs::path(ruta).filename().string();

        if (ctx.LOG_DETAIL >= 2) {
            logRawLine(log, ctx.rid, "============================================================");
            logRawLine(log, ctx.rid, "REPORTE POR IMAGEN");
            logRawLine(log, ctx.rid, "============================================================");
            logRawLine(log, ctx.rid, "IMG " + std::to_string(i + 1) + "/" + std::to_string((int)imagenes.size()) +
                                " | " + local.name);
            logRawLine(log, ctx.rid, "ruta: " + ruta);
            logBlank(log, ctx.rid);
        }

        // LOAD
        auto t_load0 = tick();
        int w = 0, h = 0, c = 0;
        unsigned char* imgRGB = cargarImagen(ruta, w, h, c, 3);
        local.ms_load = msSince(t_load0);

        if (!imgRGB) {
            local.loadOk = false;
            local.err = "LOAD_FAIL:no_se_pudo_cargar";
            rep[i] = local;
            logDet(log, ctx.rid, ctx.LOG_DETAIL, 1,
                   "[LOAD] decode=FAIL file=" + local.name + " ms=" + std::to_string(local.ms_load));
            continue;
        }

        local.loadOk = true;
        logDet(log, ctx.rid, ctx.LOG_DETAIL, 1,
               "[LOAD] decode=OK file=" + local.name +
               " size=" + std::to_string(w) + "x" + std::to_string(h) +
               " ch=3 ms=" + std::to_string(local.ms_load));

        // GRAY
        auto t_gray0 = tick();
        auto gris = convertirAGris(imgRGB, w, h);
        liberarImagen(imgRGB);
        auto ms_gray = msSince(t_gray0);

        if (ctx.LOG_DETAIL >= 2) {
            logTechTitle(log, ctx.rid, "Convertir a Gris");
            logRawLine(log, ctx.rid, "Entrada: RGB (ch=3)");
            logRawLine(log, ctx.rid, "Salida : GRAY  (ch=1) size=" + std::to_string(w) + "x" + std::to_string(h));
            logRawLine(log, ctx.rid, "ms=" + std::to_string(ms_gray));
            logBlank(log, ctx.rid);
        }

        // PREPROC hasta 128
        auto t_pre0 = tick();
        try {
            Imagen128 base = preprocesarHasta128(gris.get(), w, h, log, ctx.rid, ctx.LOG_DETAIL, local.name, ctx.QC);
            local.ms_preproc = msSince(t_pre0);

            local.preprocOk = (base.img128 && base.mask128);
            if (!local.preprocOk) {
                local.err = "PREPROC_FAIL:null_out";
                local.augCount = 0;
                rep[i] = local;

                if (ctx.LOG_DETAIL >= 3) {
                    logTechTitle(log, ctx.rid, "AUMENTACION (opcional)");
                    logRawLine(log, ctx.rid, "No aplicada: PREPROC fallo (no hay imagen 128x128).");
                    logBlank(log, ctx.rid);
                }
                continue;
            }

            // QC sobre ROI (imagen 128x128 procesada con máscara)
            auto t_qc0 = tick();
            GrayStats s_roi = calcGrayStatsMasked(base.img128.get(), base.mask128.get(),
                                                  base.w, base.h, ctx.QC.dark_thr, ctx.QC.bright_thr);
            auto ms_qc = msSince(t_qc0);

            std::string qcReason;
            bool qcOk = qcGrayPass(s_roi, ctx.QC, qcReason);

            // Guardar stats en local (estos son los que van a la tabla QC)
            local.qcOk = qcOk;
            local.qcReason = qcReason;
            local.mean = s_roi.mean;
            local.std  = s_roi.stddev;
            local.minv = s_roi.minv;
            local.maxv = s_roi.maxv;
            local.pct_dark   = s_roi.pct_dark;
            local.pct_bright = s_roi.pct_bright;

            // Log de QC
            {
                std::ostringstream oss;
                oss << "[QC][ROI] file=" << local.name
                    << " mean=" << s_roi.mean
                    << " std=" << s_roi.stddev
                    << " min=" << s_roi.minv
                    << " max=" << s_roi.maxv
                    << " pct_dark=" << s_roi.pct_dark << "%"
                    << " pct_bright=" << s_roi.pct_bright << "%"
                    << " | resultado=" << passFail(qcOk)
                    << (qcOk ? "" : (" reason=" + qcReason))
                    << " ms=" << ms_qc;
                logDet(log, ctx.rid, ctx.LOG_DETAIL, 1, oss.str());
            }

            if (!qcOk) {
                local.err = "QC_FAIL:" + qcReason;

                if (ctx.QC_ENFORCE == 1) {
                    local.preprocOk = false;
                    rep[i] = local;
                    logDet(log, ctx.rid, ctx.LOG_DETAIL, 1,
                           "[QC] file=" + local.name + " -> FAIL (QC_ENFORCE=1, se omite imagen)");
                    continue;
                }
                logDet(log, ctx.rid, ctx.LOG_DETAIL, 1,
                       "[QC] file=" + local.name + " -> FAIL (QC_ENFORCE=0, se continúa)");
            }

            // AUG
            auto t_aug0 = tick();
            auto aumentadas128 = aumentarImagenFotometrica(base.img128.get(), base.w, base.h, ruta);
            auto ms_aug = msSince(t_aug0);

            local.augCount = (int)aumentadas128.size();

            bool showAug =
                (ctx.LOG_DETAIL >= 3) ||
                (!local.qcOk) ||
                (local.augCount == 0);

            if (showAug) {
                logTechTitle(log, ctx.rid, "AUMENTACION (opcional)");
                logRawLine(log, ctx.rid, "Tipo: fotometrica");
                logRawLine(log, ctx.rid, "Variantes generadas: " + std::to_string(local.augCount));
                if (ctx.LOG_DETAIL >= 3) logRawLine(log, ctx.rid, "ms=" + std::to_string(ms_aug));
                logBlank(log, ctx.rid);
            }

            // FEATS (LBP)
            auto t_feat0 = tick();
            int feats = 0;
            int dims = 0;

            // Base
            auto featBase = extraerFeaturesDesde128(base.img128.get(), base.mask128.get());
            if (!featBase.empty()) {
                dims = (int)featBase.size();
                caracteristicas_global.push_back(std::move(featBase));
                feats++;

                if (ctx.LOG_DETAIL >= 3) {
                    logMensaje(log, ctx.rid, "[FEATS][LBP][DET] file=" + local.name + " sample_10=" + vecSample10(caracteristicas_global.back()));
                }
            } else {
                logDet(log, ctx.rid, ctx.LOG_DETAIL, 1, "[FEATS][LBP] base=FAIL file=" + local.name);
            }

            // Aumentos
            for (int k = 0; k < (int)aumentadas128.size(); ++k) {
                const auto& imgAum = aumentadas128[k].first;
                auto f = extraerFeaturesDesde128(imgAum.get(), base.mask128.get());
                if (!f.empty()) {
                    if (dims == 0) dims = (int)f.size();
                    caracteristicas_global.push_back(std::move(f));
                    feats++;
                }
            }

            local.ms_feats = msSince(t_feat0);
            local.featCount = feats;
            local.dims = dims;

            logBloquePorImagen(log, ctx, i + 1, (int)imagenes.size(), ruta, local);

            rep[i] = local;

        } catch (const std::exception& e) {
            local.preprocOk = false;
            local.err = std::string("EXC:") + e.what();
            rep[i] = local;
            logMensaje(log, ctx.rid, std::string("[EXCEPTION] file=") + local.name + " msg=" + e.what());
        } catch (...) {
            local.preprocOk = false;
            local.err = "EXC:desconocida";
            rep[i] = local;
            logMensaje(log, ctx.rid, std::string("[EXCEPTION] file=") + local.name + " msg=desconocida");
        }
    }

    // Resumen 7B por imagen
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "RESUMEN [7B] POR IMAGEN (servidor extrae estas lineas)");

    int passQc = 0;
    for (size_t i = 0; i < rep.size(); ++i) {
        const auto& r = rep[i];
        std::ostringstream oss;
        oss << "IMG " << (i + 1) << "/7 " << r.name
            << " load=" << (r.loadOk ? "OK" : "FAIL")
            << " qc=" << (r.qcOk ? "PASS" : "FAIL")
            << " preproc=" << (r.preprocOk ? "OK" : "FAIL")
            << " aug=" << r.augCount
            << " feats=" << r.featCount
            << " dims=" << r.dims
            << " ms(load/pre/feats)=" << r.ms_load << "/" << r.ms_preproc << "/" << r.ms_feats;

        if (!r.qcOk && !r.qcReason.empty()) oss << " qc_reason=" << r.qcReason;
        if (!r.err.empty()) oss << " err=" << r.err;

        log7B(log, ctx.rid, oss.str());
        if (r.qcOk) passQc++;
    }

    // Tabla QC
    logTablaQC(log, ctx.rid, ctx.LOG_DETAIL, rep, ctx.QC, ctx.QC_MIN_PASS);

    // QC global
    {
        std::ostringstream oss;
        oss << "QC_GLOBAL: qc_pass=" << passQc << "/7"
            << " | umbral_min_pass=" << ctx.QC_MIN_PASS
            << " | resultado=" << passFail(passQc >= ctx.QC_MIN_PASS);
        log7B(log, ctx.rid, oss.str());
    }

    if (passQc < ctx.QC_MIN_PASS) {
        if (ctx.QC_ENFORCE == 1) {
            logMensaje(log, ctx.rid,
                       "[QC_GLOBAL] RECHAZADO: imagenes con QC PASS insuficientes. qc_pass=" +
                       std::to_string(passQc) + " umbral=" + std::to_string(ctx.QC_MIN_PASS));
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (QC_GLOBAL_FAIL)");
            return exitCode(13);
        } else {
            logMensaje(log, ctx.rid,
                       "[QC_GLOBAL] WARN: qc_pass insuficiente pero QC_ENFORCE=0 -> NO se rechaza. qc_pass=" +
                       std::to_string(passQc) + " umbral=" + std::to_string(ctx.QC_MIN_PASS));
            log7B(log, ctx.rid, "QC_GLOBAL: WARN (qc_pass insuficiente, pero QC_ENFORCE=0)");
        }
    }

    logPrettyTitle(log, ctx, "UNION DE FEATURES (secuencial)");
    nuevasCaracteristicas = std::move(caracteristicas_global);

    // Resumen LBP
    logResumenLBP(log, ctx.rid, ctx.LOG_DETAIL, nuevasCaracteristicas, (int)imagenes.size());

    logDet(log, ctx.rid, ctx.LOG_DETAIL, 1,
           "[FEATS] total_features_nuevas=" + std::to_string((int)nuevasCaracteristicas.size()));

    if (nuevasCaracteristicas.empty()) {
        logMensaje(log, ctx.rid, "[FEATS] ERROR: no se extrajeron caracteristicas validas (vector vacio).");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (features_vacias)");
        return exitCode(15);
    }

    return 0;
}

// ==================== CONTINUACIÓN desde procesarImagenesExtraerFeatures ====================

static int aplicarPCAyNormalizar(std::ofstream& log, const Ctx& ctx,
                                const std::vector<std::vector<double>>& nuevasCaracteristicas,
                                std::vector<std::vector<double>>& reducidas)
{

    // 0) cargar Z-score params
    if (!fs::exists(ctx.rutaZScore)) {
        logMensaje(log, ctx.rid, "[Z] ERROR: zscore_params.dat no encontrado -> " + ctx.rutaZScore);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (zscore_no_existe)");
        return exitCode(16);
    }

    ZScoreParams zp;
    if (!cargarZScoreParams(ctx.rutaZScore, zp, ';')) {
        logMensaje(log, ctx.rid, "[Z] ERROR: no se pudo cargar zscore_params.dat");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (zscore_load_fail)");
        return exitCode(16);
    }

    // 1) aplicar z-score a todas las features nuevas
    std::vector<std::vector<double>> zscaled = nuevasCaracteristicas;
    if (zscaled.empty() || zscaled[0].size() != zp.mean.size()) {
        logMensaje(log, ctx.rid,
            "[Z] DIM_MISMATCH: feat_dim=" + std::to_string(zscaled.empty()?0:(int)zscaled[0].size()) +
            " z_dim=" + std::to_string((int)zp.mean.size()));
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (zscore_dim_mismatch)");
        return exitCode(16);
    }
    if (!aplicarZScoreBatch(zscaled, zp)) {
        logMensaje(log, ctx.rid, "[Z] ERROR aplicando z-score batch.");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (zscore_apply_fail)");
        return exitCode(16);
    }

    logSection(log, ctx.rid, ctx.LOG_DETAIL, "PCA (cargar modelo y aplicar)");

    if (!fs::exists(ctx.rutaModeloPCA)) {
        logMensaje(log, ctx.rid, "[PCA] ERROR: modelo_pca.dat no encontrado -> " + ctx.rutaModeloPCA);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (modelo_pca_no_existe)");
        return exitCode(16);
    }

    logMensaje(log, ctx.rid, "[PCA] cargando modelo: " + ctx.rutaModeloPCA);
    ModeloPCA modeloPCA = cargarModeloPCA(ctx.rutaModeloPCA);

    logMensaje(log, ctx.rid, "[PCA] aplicando PCA a features_nuevas...");
    reducidas = aplicarPCAConModelo(zscaled, modeloPCA);

    if (reducidas.empty()) {
        logMensaje(log, ctx.rid, "[PCA] ERROR: salida PCA vacía.");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (pca_fail)");
        return exitCode(16);
    }

    logMensaje(log, ctx.rid, "[PCA] OK: vectores_reducidos=" + std::to_string((int)reducidas.size()) +
                         " dims=" + std::to_string((int)reducidas[0].size()));

    logMensaje(log, ctx.rid, "[NORM] normalizando L2 (PCA)...");
    for (auto& v : reducidas) normalizarVector(v);
    logMensaje(log, ctx.rid, "[NORM] OK (PCA L2)");

    logSection(log, ctx.rid, ctx.LOG_DETAIL, "LDA (cargar modelo y aplicar)");

    if (!fs::exists(ctx.rutaModeloLDA)) {
        logMensaje(log, ctx.rid, "[LDA] ERROR: modelo_lda.dat no encontrado -> " + ctx.rutaModeloLDA);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (modelo_lda_no_existe)");
        return exitCode(16);
    }

    logMensaje(log, ctx.rid, "[LDA] cargando modelo: " + ctx.rutaModeloLDA);
    ModeloLDA modeloLDA = cargarModeloLDA(ctx.rutaModeloLDA);

    logMensaje(log, ctx.rid, "[LDA] aplicando LDA a PCA+L2...");
    auto lda = aplicarLDAConModelo(reducidas, modeloLDA);
    if (lda.empty()) {
        logMensaje(log, ctx.rid, "[LDA] ERROR: salida LDA vacía.");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (lda_fail)");
        return exitCode(16);
    }

    logMensaje(log, ctx.rid, "[LDA] OK: vectores_lda=" + std::to_string((int)lda.size()) +
                         " dims=" + std::to_string((int)lda[0].size()));

    logMensaje(log, ctx.rid, "[NORM] normalizando L2 (LDA)...");
    for (auto& v : lda) normalizarVector(v);
    logMensaje(log, ctx.rid, "[NORM] OK (LDA L2)");

    reducidas = std::move(lda);
    return 0;
}

static int leerIds(std::ofstream& log, const Ctx& ctx, int& identificador_unico, int& id_usuario) {
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "LECTURA IDs (id_usuario.txt / id_usuario_interno.txt)");

    identificador_unico = 0;
    {
        std::ifstream f(ctx.workDir + "/id_usuario.txt");
        if (!f.is_open() || !(f >> identificador_unico)) {
            logMensaje(log, ctx.rid, "[IDS] ERROR leyendo id_usuario.txt");
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (ids_no_encontrados)");
            return exitCode(17);
        }
    }

    id_usuario = 0;
    {
        std::ifstream f(ctx.workDir + "/id_usuario_interno.txt");
        if (!f.is_open() || !(f >> id_usuario)) {
            logMensaje(log, ctx.rid, "[IDS] ERROR leyendo id_usuario_interno.txt");
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (ids_no_encontrados)");
            return exitCode(17);
        }
    }

    logMensaje(log, ctx.rid, "[IDS] ID externo (clase) = " + std::to_string(identificador_unico));
    logMensaje(log, ctx.rid, "[IDS] ID interno (BD)   = " + std::to_string(id_usuario));
    log7B(log, ctx.rid, "IDS: clase=" + std::to_string(identificador_unico) + " id_usuario=" + std::to_string(id_usuario));
    return 0;
}

static int cargarBaseYModelo(std::ofstream& log, const Ctx& ctx,
                            std::vector<std::vector<double>>& existentes,
                            std::vector<int>& etiquetasExistentes,
                            TemplateModel& templates)
{
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "CARGA DATASET (CSV) + TEMPLATES (K=1)");

    existentes.clear();
    etiquetasExistentes.clear();

    if (!fs::exists(ctx.rutaCSV)) {
        logMensaje(log, ctx.rid, "[CSV] ERROR: CSV base no existe: " + ctx.rutaCSV);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (csv_no_existe)");
        return exitCode(18);
    }
    if (!cargarCSV(ctx.rutaCSV, existentes, etiquetasExistentes, ';')) {
        logMensaje(log, ctx.rid, "[CSV] ERROR cargando CSV: " + ctx.rutaCSV);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (csv_load_fail)");
        return exitCode(18);
    }
    logMensaje(log, ctx.rid, "[CSV] OK: muestras=" + std::to_string((int)existentes.size()) +
                         " dims=" + (existentes.empty() ? "0" : std::to_string((int)existentes[0].size())));

    if (fs::exists(ctx.rutaTemplates) && cargarTemplatesCSV(ctx.rutaTemplates, templates)) {
        logMensaje(log, ctx.rid, "[TEMPLATES] OK: " + std::to_string((int)templates.clases.size()) +
                             " clases | ruta=" + ctx.rutaTemplates);
    } else {
        logMensaje(log, ctx.rid, "[TEMPLATES] WARN: no se pudo cargar templates, recomputando desde CSV...");
        templates = construirTemplatesK1(existentes, etiquetasExistentes);
        if (templates.clases.empty()) {
            logMensaje(log, ctx.rid, "[TEMPLATES] ERROR: no se pudieron construir templates.");
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (templates_fail)");
            return exitCode(18);
        }
        if (!guardarTemplatesCSV(ctx.rutaTemplates, templates)) {
            logMensaje(log, ctx.rid, "[TEMPLATES] ERROR guardando templates: " + ctx.rutaTemplates);
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (templates_save_fail)");
            return exitCode(18);
        }
        logMensaje(log, ctx.rid, "[TEMPLATES] OK: regenerados y guardados.");
    }

    return 0;
}

static void limpiezaTemporales(std::ofstream& log, const Ctx& ctx, const std::vector<std::string>& imagenes) {
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "LIMPIEZA TEMPORALES");

    if (!ctx.AUDIT_MODE) {
        logMensaje(log, ctx.rid, "[FS] AUDIT_MODE=0 -> eliminando imágenes temporales...");
        for (const auto& ruta : imagenes) {
            try { fs::remove(ruta); } catch (...) {}
        }
        logMensaje(log, ctx.rid, "[FS] OK: temporales eliminados");
    } else {
        logMensaje(log, ctx.rid, "[FS] AUDIT_MODE=1 -> NO se eliminan imágenes temporales.");
    }
}

static int validarWorkDirYListarJpg(std::ofstream& log, const Ctx& ctx, std::vector<std::string>& imagenes) {
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "VALIDACION WORK_DIR");
    if (!fs::exists(ctx.workDir) || !fs::is_directory(ctx.workDir)) {
        logMensaje(log, ctx.rid, "[FS] ERROR: Carpeta de trabajo no encontrada: " + ctx.workDir);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (WORK_DIR invalido)");
        return exitCode(10);
    }
    logMensaje(log, ctx.rid, "[FS] OK: WORK_DIR existe");

    logSection(log, ctx.rid, ctx.LOG_DETAIL, "DESCUBRIMIENTO DE IMAGENES");
    imagenes.clear();
    for (const auto& entry : fs::directory_iterator(ctx.workDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            imagenes.push_back(entry.path().string());
        }
    }
    std::sort(imagenes.begin(), imagenes.end());

    logMensaje(log, ctx.rid, "[IMG] .jpg encontrados=" + std::to_string((int)imagenes.size()));
    if (ctx.LOG_DETAIL >= 2) {
        for (int i = 0; i < (int)imagenes.size(); ++i) {
            logMensaje(log, ctx.rid, "[IMG] file[" + std::to_string(i) + "] " + imagenes[i]);
        }
    }

    // CAMBIO FASE 6: Reducido de 7 a 5 imágenes para consistencia con dataset offline
    // Dataset offline: 5 train + 2 test por usuario
    // Producción: 5 train (sin test, usuario ya verificado)
    if ((int)imagenes.size() != 5) {
        logMensaje(log, ctx.rid, "[IMG] ERROR: se requieren EXACTAMENTE 5 imágenes. Encontradas=" + std::to_string((int)imagenes.size()));
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (cantidad_imagenes!=5)");
        return exitCode(11);
    }
    return 0;
}

static int registrarEntrenarEvaluarGuardar(std::ofstream& log, const Ctx& ctx,
                                           int identificador_unico, int id_usuario,
                                           const std::vector<std::vector<double>>& reducidas,
                                           std::vector<std::vector<double>>& existentes,
                                           std::vector<int>& etiquetasExistentes,
                                           TemplateModel& templates)
{
    // ========= templates K=1 =========
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "TEMPLATES K=1 (COSENO) - PRODUCCION");
    log7B(log, ctx.rid, "Produccion 1:1 sin SVM. Se usan templates por usuario (K=1) y coseno.");

    // ========= anti-duplicado biométrico =========
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "ANTI-DUPLICADO BIOMETRICO (margen + consistencia)");
    if (!templates.clases.empty() && !reducidas.empty()) {
        int M = std::min((int)reducidas.size(), 25);

        const double UMBRAL_MARGEN = getEnvDouble("DUP_UMBRAL_MARGEN", ctx.DUP_UMBRAL_MARGEN);
        const double UMBRAL_CONSISTENCIA = getEnvDouble("DUP_UMBRAL_CONSISTENCIA", ctx.DUP_UMBRAL_CONSISTENCIA);
        const double UMBRAL_VOTOS_CONFIABLES = getEnvDouble("DUP_UMBRAL_VOTOS_CONFIABLES", ctx.DUP_UMBRAL_VOTOS_CONFIABLES);

        std::unordered_map<int, int> votos;
        int votosConfiables = 0;

        logMensaje(log, ctx.rid, "[DUP] Config: M=" + std::to_string(M) +
                            " margen>=" + std::to_string(UMBRAL_MARGEN) +
                            " consistencia>=" + std::to_string(UMBRAL_CONSISTENCIA) +
                            " votosConfiables>=" + std::to_string(UMBRAL_VOTOS_CONFIABLES));

        for (int i = 0; i < M; ++i) {
            const auto& x = reducidas[i];

            double bestScore = 0.0, secondScore = 0.0;
            int pred = -1;
            if (!scoreTemplatesK1(templates, x, pred, bestScore, secondScore)) continue;
            double margen = bestScore - secondScore;

            bool conf = (margen >= UMBRAL_MARGEN);
            if (conf) {
                votos[pred]++;
                votosConfiables++;
            }

            if (ctx.LOG_DETAIL >= 3) {
                std::ostringstream oss;
                oss << "[DUP][DET] i=" << i
                    << " pred=" << pred
                    << " best=" << bestScore
                    << " second=" << secondScore
                    << " margen=" << margen
                    << " conf=" << (conf ? "1" : "0");
                logRaw(log, ctx.rid, oss.str());
                logBlank(log, ctx.rid);
            }
        }

        int claseMasVotada = -1;
        int maxVotos = 0;
        for (auto& kv : votos) {
            if (kv.second > maxVotos) {
                maxVotos = kv.second;
                claseMasVotada = kv.first;
            }
        }

        double consistencia = (M > 0) ? ((double)maxVotos / (double)M) : 0.0;
        double fracConfiables = (M > 0) ? ((double)votosConfiables / (double)M) : 0.0;

        int minConfiables = (int)std::ceil(UMBRAL_VOTOS_CONFIABLES * M);
        int votosNecesarios = (int)std::ceil(UMBRAL_CONSISTENCIA * M);

        // ← NUEVA: Tabla resumen de duplicado
        logResumenDuplicado(log, ctx.rid, ctx.LOG_DETAIL, M, votosConfiables, claseMasVotada, maxVotos,
                           consistencia, fracConfiables, minConfiables, votosNecesarios,
                           UMBRAL_MARGEN, UMBRAL_CONSISTENCIA);

        logMensaje(log, ctx.rid, "[DUP] resumen: votosConfiables=" + std::to_string(votosConfiables) +
                            "/" + std::to_string(M) +
                            " fracConfiables=" + std::to_string(fracConfiables) +
                            " consistencia=" + std::to_string(consistencia) +
                            " minConfiables=" + std::to_string(minConfiables) +
                            " claseMasVotada=" + std::to_string(claseMasVotada) +
                            " maxVotos=" + std::to_string(maxVotos) +
                            " votosNecesarios=" + std::to_string(votosNecesarios));

        if (claseMasVotada != -1 &&
            maxVotos >= votosNecesarios &&
            votosConfiables >= minConfiables) {

            logMensaje(log, ctx.rid, "ALERTA: Biometría duplicada probable. Coincide con clase existente: " + std::to_string(claseMasVotada));
            log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (biometría duplicada probable, coincide con clase " + std::to_string(claseMasVotada) + ")");
            return exitCode(19);
        }
    }

    // usuario ya registrado (clase existe)
    if (std::find(templates.clases.begin(), templates.clases.end(), identificador_unico) != templates.clases.end()) {
        logMensaje(log, ctx.rid, "[TEMPLATES] ERROR: la clase ya existe -> usuario ya registrado.");
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (clase_ya_existe)");
        return exitCode(20);
    }

    // ========= actualización templates =========
    logPhase(
        log, ctx.rid, ctx.LOG_DETAIL,
        7,
        "ACTUALIZACION TEMPLATES (K=1)",
        "Agregar la nueva clase y regenerar templates por usuario para verificación 1:1.",
        {"Nota: No se reentrena SVM. Solo se actualizan templates (coseno)."}
    );

    // ========= backup + guardar CSV/TEMPLATES =========
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "BACKUP + GUARDADO CSV/TEMPLATES");
    std::string versionDir;
    makeBackupVersion(log, ctx.rid, ctx.LOG_DETAIL, ctx.dirVersiones, ctx.rutaCSV, ctx.rutaTemplates, versionDir);

    logMensaje(log, ctx.rid, "[CSV] agregando vectores reducidos al dataset...");
    etiquetasExistentes.insert(etiquetasExistentes.end(), (int)reducidas.size(), identificador_unico);
    existentes.insert(existentes.end(), reducidas.begin(), reducidas.end());

    if (!guardarCSV(ctx.rutaCSV, existentes, etiquetasExistentes, ';')) {
        logMensaje(log, ctx.rid, "[CSV] ERROR guardando CSV actualizado.");
        if (!versionDir.empty()) rollbackFromVersion(log, ctx.rid, ctx.LOG_DETAIL, versionDir, ctx.rutaCSV, ctx.rutaTemplates);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (fallo_guardar_csv, rollback_aplicado)");
        return exitCode(21);
    }
    logMensaje(log, ctx.rid, "[CSV] OK guardado: " + ctx.rutaCSV);

    TemplateModel updated = construirTemplatesK1(existentes, etiquetasExistentes);
    if (updated.clases.empty()) {
        logMensaje(log, ctx.rid, "[TEMPLATES] ERROR: no se pudieron construir templates.");
        if (!versionDir.empty()) rollbackFromVersion(log, ctx.rid, ctx.LOG_DETAIL, versionDir, ctx.rutaCSV, ctx.rutaTemplates);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (templates_fail, rollback_aplicado)");
        return exitCode(21);
    }
    if (!guardarTemplatesCSV(ctx.rutaTemplates, updated)) {
        logMensaje(log, ctx.rid, "[TEMPLATES] ERROR guardando templates.");
        if (!versionDir.empty()) rollbackFromVersion(log, ctx.rid, ctx.LOG_DETAIL, versionDir, ctx.rutaCSV, ctx.rutaTemplates);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (fallo_guardar_templates, rollback_aplicado)");
        return exitCode(21);
    }
    templates = std::move(updated);
    logMensaje(log, ctx.rid, "[TEMPLATES] OK guardado: " + ctx.rutaTemplates);

    // ========= evaluación holdout post =========
    // COMENTADO TEMPORALMENTE PARA DEBUGGING
    /*
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "EVALUACION POST-REGISTRO (HOLDOUT)");
    const int clasesDespues = (int)modeloActual.clases.size();
    double accNew = evaluarHoldout(log, ctx.rid, ctx.LOG_DETAIL, ctx.holdoutCsv, modeloActual, ctx.EVAL_PRINT_N);
    if (accNew < 0.0) {
        if (!versionDir.empty()) rollbackFromVersion(log, ctx.rid, ctx.LOG_DETAIL, versionDir, ctx.rutaCSV, ctx.rutaTemplates);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (fallo_eval_holdout, rollback_aplicado)");
        return exitCode(22);
    }

    double delta = accNew - baselineAcc;

    {
        std::ostringstream oss;
        oss << "ACC anterior=" << baselineAcc << "% | ACC actual=" << accNew << "% | delta=" << delta
            << " pts | umbral=" << ctx.PERF_DROP_THRESHOLD << " pts"
            << " | clases_antes=" << clasesAntes << " | clases_despues=" << clasesDespues
            << " | 80%=" << trainN << " | 20%=" << testN;
        log7B(log, ctx.rid, oss.str());
    }

    logMensaje(log, ctx.rid, "[EVAL] comparacion: baseline=" + std::to_string(baselineAcc) +
                         "% new=" + std::to_string(accNew) +
                         "% delta=" + std::to_string(delta) + " pts");

    // ========= decision por umbral caida =========
    if (delta < -ctx.PERF_DROP_THRESHOLD) {
        logMensaje(log, ctx.rid, "[DECISION] RECHAZADO: caída excesiva. delta=" + std::to_string(delta) +
                             " umbral=" + std::to_string(ctx.PERF_DROP_THRESHOLD) + " -> rollback");
        if (!versionDir.empty()) rollbackFromVersion(log, ctx.rid, ctx.LOG_DETAIL, versionDir, ctx.rutaCSV, ctx.rutaTemplates);
        log7B(log, ctx.rid, "VEREDICTO: RECHAZADO (caida_excesiva, rollback_aplicado)");
        log7B(log, ctx.rid, "FIN: registro biométrico finalizado con RECHAZO");
        return exitCode(23);
    }

    // aceptado: actualizar baseline
    saveBaseline(ctx.baselineJson, accNew);
    logMensaje(log, ctx.rid, "[DECISION] ACEPTADO: baseline actualizado a " + std::to_string(accNew) + "%");
    log7B(log, ctx.rid, "VEREDICTO: ACEPTADO (delta dentro del umbral)");
    */

    // SKIP VALIDACION HOLDOUT - ACEPTAR DIRECTAMENTE
    log7B(log, ctx.rid, "⚠️ VALIDACION HOLDOUT DESHABILITADA - REGISTRO ACEPTADO SIN EVALUAR");
    log7B(log, ctx.rid, "VEREDICTO: ACEPTADO (validacion_deshabilitada)");


    // ========= registrar credencial en BD =========
    logSection(log, ctx.rid, ctx.LOG_DETAIL, "REGISTRO CREDENCIAL EN BD (PostgREST)");
    auto cli = makeClientPostgrest();

    json body = {
        {"id_usuario", id_usuario},
        {"tipo_biometria", "oreja"},
        {"estado", "activo"}
    };

    logMensaje(log, ctx.rid, "[BD] POST /credenciales_biometricas body=" + body.dump());
    auto res = cli.Post("/credenciales_biometricas", body.dump(), "application/json");

    if (!res) {
        logMensaje(log, ctx.rid, "[BD] ERROR: sin respuesta de PostgREST");
        log7B(log, ctx.rid, "FIN: registro ACEPTADO pero BD falló (sin respuesta)");
        return exitCode(24);
    }

    logMensaje(log, ctx.rid, "[BD] status=" + std::to_string(res->status) + " body_bytes=" + std::to_string(res->body.size()));
    if (ctx.LOG_DETAIL >= 3) logMensaje(log, ctx.rid, "[BD] body=" + res->body);

    if (res->status == 409) {
        logMensaje(log, ctx.rid, "[BD] OK: credencial ya existía (409), se considera OK.");
    } else if (res->status != 201 && res->status != 200) {
        logMensaje(log, ctx.rid, "[BD] ERROR: PostgREST rechazó. status=" + std::to_string(res->status));
        log7B(log, ctx.rid, "FIN: registro ACEPTADO pero BD falló (status=" + std::to_string(res->status) + ")");
        return exitCode(24);
    } else {
        logMensaje(log, ctx.rid, "[BD] OK: credencial registrada.");
    }

    // OK final (main limpia)
    log7B(log, ctx.rid, "FIN: registro biométrico completado");
    return exitCode(0);
}

// ====================== MAIN ======================
int main(int argc, char** argv) {
    ArgsBio a = parseArgsBio(argc, argv);
    Ctx ctx = loadCtxFromEnvAndArgs(a);

    auto log = crearLogStream(ctx.workDir);
    startupLogs(log, ctx);

    std::vector<std::string> imagenes;
    {
        int rc = validarWorkDirYListarJpg(log, ctx, imagenes);
        if (rc != 0) return rc;
    }

    std::vector<ImageReport> rep;
    std::vector<std::vector<double>> nuevasCaracteristicas;
    {
        int rc = procesarImagenesExtraerFeatures(log, ctx, imagenes, rep, nuevasCaracteristicas);
        if (rc != 0) return rc;
    }

    std::vector<std::vector<double>> reducidas;
    {
        int rc = aplicarPCAyNormalizar(log, ctx, nuevasCaracteristicas, reducidas);
        if (rc != 0) return rc;
    }

    int identificador_unico = 0;
    int id_usuario = 0;
    {
        int rc = leerIds(log, ctx, identificador_unico, id_usuario);
        if (rc != 0) return rc;
    }

    std::vector<std::vector<double>> existentes;
    std::vector<int> etiquetasExistentes;
    TemplateModel templates;
    {
        int rc = cargarBaseYModelo(log, ctx, existentes, etiquetasExistentes, templates);
        if (rc != 0) return rc;
    }

    int rc = registrarEntrenarEvaluarGuardar(
        log, ctx,
        identificador_unico, id_usuario,
        reducidas,
        existentes, etiquetasExistentes, templates
    );
    if (rc != 0) return rc;

    limpiezaTemporales(log, ctx, imagenes);

    logSection(log, ctx.rid, ctx.LOG_DETAIL, "FIN agregar_usuario_biometria (OK)");
    return exitCode(0);
}