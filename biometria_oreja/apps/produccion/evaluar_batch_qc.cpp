#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <memory>
#include <cctype>

// === includes sincronizados con procesar_dataset.cpp ===
#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/normalizacion.h"
#include "utilidades/pca_utils.h"
#include "utilidades/svm_ova_utils.h"
#include "utilidades/zscore_params.h"
#include "svm/svm_prediccion.h"

namespace fs = std::filesystem;

static long long ms_since(const std::chrono::steady_clock::time_point& t0) {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now() - t0).count();
}

// ====== Reuso literal de tu pipeline ======
struct Imagen128 {
    std::unique_ptr<uint8_t[]> img128;
    std::unique_ptr<uint8_t[]> mask128;
    int w = 128;
    int h = 128;
};

static Imagen128 preprocesarHasta128(const uint8_t* imagenGris, int ancho, int alto) {
    // ============================================================================
    // PIPELINE FASE 6 - SINCRONIZADO CON procesar_dataset.cpp
    // ============================================================================
    // 1. Resize directo a 128x128
    // 2. CLAHE (8×8 tiles, clipLimit=2.0)
    // 3. Bilateral (σ_space=3, σ_color=50)
    // 4. Máscara elíptica FIJA
    // ============================================================================

    Imagen128 out;

    auto img128 = redimensionarParaBiometria(imagenGris, ancho, alto, 128, 128);
    auto img128_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);
    out.img128 = aplicarBilateral(img128_clahe.get(), 128, 128, 3.0, 50.0);
    out.mask128 = crearMascaraElipticaFija(128, 128);

    return out;
}

static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    // SINCRONIZADO con procesar_dataset.cpp: 6x6 bloques, threshold=200
    // Multi-Scale LBP: 6×6×118 = 4248 dimensions
    // ✅ Multi-Scale LBP
    return calcularLBPMultiEscalaPorBloquesRobustoNorm(img128, mask128, 128, 128, 6, 6, 200, true);
}

static std::vector<double> extraerCaracteristicas(const uint8_t* imagenGris, int ancho, int alto) {
    Imagen128 base = preprocesarHasta128(imagenGris, ancho, alto);
    return extraerFeaturesDesde128(base.img128.get(), base.mask128.get());
}

static bool predecirTop1Top2(
    const std::vector<double>& x,
    const ModeloSVM& modelo,
    int& claseTop1,
    double& scoreTop1,
    double& scoreTop2
) {
    if (modelo.clases.empty() || modelo.pesosPorClase.empty() || modelo.biasPorClase.empty())
        return false;

    scoreTop1 = -std::numeric_limits<double>::infinity();
    scoreTop2 = -std::numeric_limits<double>::infinity();
    claseTop1 = -1;

    for (int i = 0; i < (int)modelo.clases.size(); ++i) {
        const auto& w = modelo.pesosPorClase[i];
        double b = modelo.biasPorClase[i];
        if (w.size() != x.size()) continue;

        double s = std::inner_product(x.begin(), x.end(), w.begin(), 0.0) + b;

        if (s > scoreTop1) {
            scoreTop2 = scoreTop1;
            scoreTop1 = s;
            claseTop1 = modelo.clases[i];
        } else if (s > scoreTop2) {
            scoreTop2 = s;
        }
    }
    return (claseTop1 != -1) && std::isfinite(scoreTop1) && std::isfinite(scoreTop2);
}

// ====== QC mínimo (si ya tienes uno, reemplaza por tu función real) ======
struct GrayStats {
    double mean = 0.0;
    double stddev = 0.0;
    double pct_dark = 0.0;
    double pct_bright = 0.0;
    int minv = 255;
    int maxv = 0;
};

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

// ====== Parse mapa_etiquetas.txt ======
static bool cargarMapaEtiquetas(const std::string& path,
                               std::unordered_map<int,int>& real2internal) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::regex re(R"(Etiqueta\s+real\s+(\d+)\s+\?\s+clase\s+interna\s+(\d+))");
    std::string line;
    while (std::getline(f, line)) {
        std::smatch m;
        if (std::regex_search(line, m, re)) {
            int real = std::stoi(m[1].str());
            int internal = std::stoi(m[2].str());
            real2internal[real] = internal;
        }
    }
    return !real2internal.empty();
}

// ====== Condiciones (aplicadas sobre imagen gris) ======
enum class Cond { Base, BrightP20, BrightM20, Contrast1p1, Gamma0p9, Gamma1p1, NoiseP10, NoiseM10 };

static const char* cond_name(Cond c) {
    switch (c) {
        case Cond::Base: return "base";
        case Cond::BrightP20: return "bright_p20";
        case Cond::BrightM20: return "bright_m20";
        case Cond::Contrast1p1: return "contrast_1p1";
        case Cond::Gamma0p9: return "gamma_0p9";
        case Cond::Gamma1p1: return "gamma_1p1";
        case Cond::NoiseP10: return "noise_p10";
        case Cond::NoiseM10: return "noise_m10";
    }
    return "unknown";
}

static inline int clamp255(int v) { return (v<0?0:(v>255?255:v)); }

static std::unique_ptr<uint8_t[]> apply_condition(const uint8_t* gray, int w, int h, Cond c) {
    const int n = w*h;
    auto out = std::make_unique<uint8_t[]>(n);

    // params
    const int bright_delta_p = 20;
    const int bright_delta_m = -20;
    const double contrast = 1.1;

    auto apply_gamma = [&](double g){
        for (int i=0;i<n;++i) {
            double x = gray[i] / 255.0;
            double y = std::pow(x, g);
            out[i] = (uint8_t)clamp255((int)std::round(y * 255.0));
        }
    };

    switch (c) {
        case Cond::Base:
            std::copy(gray, gray+n, out.get());
            break;
        case Cond::BrightP20:
            for (int i=0;i<n;++i) out[i] = (uint8_t)clamp255((int)gray[i] + bright_delta_p);
            break;
        case Cond::BrightM20:
            for (int i=0;i<n;++i) out[i] = (uint8_t)clamp255((int)gray[i] + bright_delta_m);
            break;
        case Cond::Contrast1p1:
            for (int i=0;i<n;++i) {
                int v = (int)std::round((gray[i] - 128) * contrast + 128);
                out[i] = (uint8_t)clamp255(v);
            }
            break;
        case Cond::Gamma0p9: apply_gamma(0.9); break;
        case Cond::Gamma1p1: apply_gamma(1.1); break;
        case Cond::NoiseP10:
        case Cond::NoiseM10: {
            // ruido simple +/-10 (uniforme). Reproducible con semilla fija.
            // Nota: para “noise_p10” y “noise_m10” lo hacemos simétrico, solo cambia el signo base.
            // Si tú ya tienes una función de ruido, reemplaza esto por tu implementación.
            uint32_t seed = 1337u;
            auto rng = [&](){
                seed = 1664525u * seed + 1013904223u;
                return seed;
            };
            int sign = (c == Cond::NoiseP10) ? +1 : -1;
            for (int i=0;i<n;++i) {
                int r = (int)(rng() % 11); // 0..10
                int v = (int)gray[i] + sign * r;
                out[i] = (uint8_t)clamp255(v);
            }
            break;
        }
    }
    return out;
}

static int parse_user_real_from_filename(const std::string& fn) {
    if (fn.size() < 3) return -1;
    return std::stoi(fn.substr(0,3));
}

int main(int argc, char** argv) {
    std::string dataset = "";
    std::string outdir = "out";
    std::string csv = "resultados_batch_qc.csv";

    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a == "--dataset" && i+1<argc) dataset = argv[++i];
        else if (a == "--out" && i+1<argc) outdir = argv[++i];
        else if (a == "--csv" && i+1<argc) csv = argv[++i];
    }
    if (dataset.empty()) {
        std::cerr << "Uso: evaluar_batch_qc --dataset <carpeta_subset_100> [--out <outdir>] [--csv <salida.csv>]\n";
        return 1;
    }

    // 1) cargar mapping real->internal
    std::unordered_map<int,int> real2internal;
    std::string path_map = (fs::path(outdir) / "mapa_etiquetas.txt").string();
    if (!cargarMapaEtiquetas(path_map, real2internal)) {
        std::cerr << "No pude cargar mapa_etiquetas.txt en: " << path_map << "\n";
        return 2;
    }

    // 2) cargar PCA y SVM
    std::string path_pca = (fs::path(outdir) / "modelo_pca.dat").string();
    std::string path_svm = (fs::path(outdir) / "modelo_svm.svm").string();

    if (!fs::exists(path_pca) || !fs::exists(path_svm)) {
        std::cerr << "Faltan modelos en outdir. PCA=" << path_pca << " SVM=" << path_svm << "\n";
        return 3;
    }

    ModeloPCA pca = cargarModeloPCA(path_pca);
    ModeloSVM svm;
    if (!cargarModeloSVM(path_svm, svm)) {
        std::cerr << "Error cargando SVM desde: " << path_svm << "\n";
        return 4;
    }

    // 2.5) cargar Z-score params (CRÍTICO - consistencia con entrenamiento)
    std::string path_zscore = (fs::path(outdir) / "zscore_params.dat").string();
    ZScoreParams zp;
    if (!fs::exists(path_zscore) || !cargarZScoreParams(path_zscore, zp, ';')) {
        std::cerr << "Error: Z-score params NO disponibles en: " << path_zscore << "\n";
        return 44;
    }
    std::cerr << "Z-score params cargados OK (dim=" << zp.mean.size() << ")\n";

    // 3) abrir CSV
    std::ofstream f(csv);
    if (!f.is_open()) {
        std::cerr << "No pude abrir CSV salida: " << csv << "\n";
        return 5;
    }
    f << "user_real,y_true,filename,condition,"
         "qc_mean,qc_std,qc_pct_dark,qc_pct_bright,qc_min,qc_max,"
         "pred,score1,score2,margin,ok,time_ms\n";

    // 4) recorrer imágenes
    std::vector<Cond> conds = {
        Cond::Base, Cond::BrightP20, Cond::BrightM20, Cond::Contrast1p1,
        Cond::Gamma0p9, Cond::Gamma1p1, Cond::NoiseP10, Cond::NoiseM10
    };

    int count_files = 0;
    for (auto& entry : fs::directory_iterator(dataset)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        auto ext = path.extension().string();
        for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        std::string fn = path.filename().string();
        int user_real = parse_user_real_from_filename(fn);
        if (user_real < 0) continue;

        auto it = real2internal.find(user_real);
        if (it == real2internal.end()) continue;
        int y_true = it->second;

        // cargar imagen RGB
        int w=0,h=0,ch=0;
        unsigned char* rgb = cargarImagen(path.string().c_str(), w, h, ch, 3);
        if (!rgb) continue;

        // gris base
        auto gray_base = convertirAGris(rgb, w, h);
        delete[] rgb;

        for (auto c : conds) {
            auto t0 = std::chrono::steady_clock::now();

            /// aplicar condición sobre gris
            auto gray = apply_condition(gray_base.get(), w, h, c);

            // QC (idéntico a agregar_usuario_biometria)
            GrayStats qc = calcGrayStats(gray.get(), w, h);

            // features LBP
            auto feat = extraerCaracteristicas(gray.get(), w, h);
            if (feat.empty()) continue;

            // Z-score (CRÍTICO - consistencia con entrenamiento)
            if (feat.size() != zp.mean.size()) {
                std::cerr << "ERROR: dim mismatch feat=" << feat.size()
                          << " zscore=" << zp.mean.size() << "\n";
                continue;
            }
            if (!aplicarZScore(feat, zp)) continue;

            // PCA
            auto red = aplicarPCAConModelo({ feat }, pca);
            if (red.empty() || red[0].empty()) continue;

            // Normalización L2 ELIMINADA (procesar_dataset.cpp NO la usa)
            // El SVM fue entrenado con vectores SIN normalización L2 post-PCA
            // normalizarVector(red[0]);  // ← COMENTADO: causaba predicciones incorrectas

            int pred=-1;
            double s1=0, s2=0;
            if (!predecirTop1Top2(red[0], svm, pred, s1, s2)) continue;

            double margin = s1 - s2;
            int ok = (pred == y_true) ? 1 : 0;

            long long ms = ms_since(t0);

            f << user_real << "," << y_true << ","
            << "\"" << fn << "\"" << ","
            << cond_name(c) << ","
            << qc.mean << "," << qc.stddev << ","
            << qc.pct_dark << "," << qc.pct_bright << ","
            << qc.minv << "," << qc.maxv << ","
            << pred << "," << s1 << "," << s2 << ","
            << margin << "," << ok << "," << ms << "\n";
        }

        count_files++;
    }

    std::cerr << "OK. Archivos procesados=" << count_files << " CSV=" << csv << "\n";
    return 0;
}