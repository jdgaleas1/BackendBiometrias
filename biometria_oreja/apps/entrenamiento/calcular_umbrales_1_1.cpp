//
// calcular_umbrales_1_1.cpp
// Calcula umbrales óptimos para autenticación 1:1 (verificación biométrica)
//
// Para cada muestra de test:
//   - Score genuino: score de su clase real
//   - Scores impostores: scores de todas las demás clases
//
// Calcula FAR/FRR para diferentes umbrales y encuentra:
//   - Umbral EER (donde FAR = FRR)
//   - Umbrales para diferentes puntos de operación
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <memory>

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

// ====== Pipeline FASE 6 (sincronizado con procesar_dataset.cpp) ======
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
    // 1. Resize directo a 128x128 (SIN bilateral previo, SIN detección de región)
    // 2. CLAHE (8×8 tiles, clipLimit=2.0) - Mejora contraste local
    // 3. Bilateral (σ_space=3, σ_color=50) - Reduce ruido post-CLAHE
    // 4. Máscara elíptica FIJA (consistente entre todas las imágenes)
    // ============================================================================

    Imagen128 out;

    // Paso 1: Resize directo a 128x128
    auto img128 = redimensionarParaBiometria(imagenGris, ancho, alto, 128, 128);

    // Paso 2: CLAHE (8×8 tiles, clipLimit=2.0)
    auto img128_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);

    // Paso 3: Bilateral Filter (σ_space=3, σ_color=50)
    out.img128 = aplicarBilateral(img128_clahe.get(), 128, 128, 3.0, 50.0);

    // Paso 4: Máscara elíptica FIJA
    out.mask128 = crearMascaraElipticaFija(128, 128);

    return out;
}

static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    // LBP Multi-Scale (radius=1 + radius=2): 6x6 bloques, 200 umbral
    // IMPORTANTE: Debe coincidir EXACTAMENTE con procesar_dataset.cpp
    // Dimensiones: 6×6 bloques × 118 bins (multi-scale) = 4248 features
    return calcularLBPMultiEscalaPorBloquesRobustoNorm(img128, mask128, 128, 128, 6, 6, 200, true);
}

static std::vector<double> extraerCaracteristicas(const uint8_t* imagenGris, int ancho, int alto) {
    Imagen128 base = preprocesarHasta128(imagenGris, ancho, alto);
    return extraerFeaturesDesde128(base.img128.get(), base.mask128.get());
}

// Calcula TODOS los scores para una muestra
static std::vector<double> calcularTodosLosScores(
    const std::vector<double>& x,
    const ModeloSVM& modelo
) {
    std::vector<double> scores;
    scores.reserve(modelo.clases.size());

    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        const auto& w = modelo.pesosPorClase[i];
        double b = modelo.biasPorClase[i];

        double s = std::inner_product(x.begin(), x.end(), w.begin(), 0.0) + b;
        scores.push_back(s);
    }

    return scores;
}

struct MetricasVerificacion {
    double umbral;
    double far;  // False Accept Rate
    double frr;  // False Reject Rate
    double err;  // Equal Error Rate (si FAR ≈ FRR)
};

int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "  CÁLCULO DE UMBRALES ÓPTIMOS PARA 1:1\n";
    std::cout << "==============================================\n\n";

    std::string dataset_test = "test_norm";
    std::string outdir = "out";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--test" && i + 1 < argc) dataset_test = argv[++i];
        else if (a == "--out" && i + 1 < argc) outdir = argv[++i];
    }

    if (!fs::exists(dataset_test)) {
        std::cerr << "ERROR: dataset_test no existe: " << dataset_test << "\n";
        return 1;
    }

    // 1) Cargar modelos
    std::string path_pca = (fs::path(outdir) / "modelo_pca.dat").string();
    std::string path_svm = (fs::path(outdir) / "modelo_svm.svm").string();

    if (!fs::exists(path_pca) || !fs::exists(path_svm)) {
        std::cerr << "ERROR: Faltan modelos. PCA=" << path_pca << " SVM=" << path_svm << "\n";
        return 2;
    }

    ModeloPCA pca = cargarModeloPCA(path_pca);
    ModeloSVM svm;
    if (!cargarModeloSVM(path_svm, svm)) {
        std::cerr << "ERROR: No se pudo cargar SVM\n";
        return 3;
    }

    // 1.5) Cargar Z-score params (CRÍTICO - consistencia con entrenamiento)
    std::string path_zscore = (fs::path(outdir) / "zscore_params.dat").string();
    ZScoreParams zp;
    if (!fs::exists(path_zscore) || !cargarZScoreParams(path_zscore, zp, ';')) {
        std::cerr << "ERROR: Z-score params NO disponibles en: " << path_zscore << "\n";
        return 33;
    }

    std::cout << "Modelos cargados:\n";
    std::cout << "  - PCA: " << pca.componentes.size() << " componentes\n";
    std::cout << "  - SVM: " << svm.clases.size() << " clases\n";
    std::cout << "  - Z-score: " << zp.mean.size() << " dimensiones\n\n";

    // 2) Procesar test y recolectar scores
    std::vector<double> scores_genuinos;
    std::vector<double> scores_impostores;

    int num_muestras = 0;
    int errores = 0;

    std::cout << "Procesando dataset de test...\n";

    for (const auto& entry : fs::directory_iterator(dataset_test)) {
        if (!entry.is_regular_file()) continue;

        auto path = entry.path();
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        // Extraer etiqueta real del nombre del archivo (formato: XXX_*.jpg)
        std::string filename = path.filename().string();
        if (filename.size() < 3) continue;

        int etiqueta_real = -1;
        try {
            etiqueta_real = std::stoi(filename.substr(0, 3));
        } catch (...) {
            continue;
        }

        // Buscar índice de la clase real en el modelo
        auto it = std::find(svm.clases.begin(), svm.clases.end(), etiqueta_real);
        if (it == svm.clases.end()) {
            std::cerr << "WARN: Clase " << etiqueta_real << " no está en el modelo\n";
            continue;
        }
        int idx_clase_real = (int)std::distance(svm.clases.begin(), it);

        // Cargar y procesar imagen
        int w = 0, h = 0, ch = 0;
        unsigned char* rgb = cargarImagen(path.string().c_str(), w, h, ch, 3);
        if (!rgb) {
            errores++;
            continue;
        }

        auto gris = convertirAGris(rgb, w, h);
        delete[] rgb;

        auto feat = extraerCaracteristicas(gris.get(), w, h);
        if (feat.empty()) {
            errores++;
            continue;
        }

        // Z-score (CRÍTICO - consistencia con entrenamiento)
        if (feat.size() != zp.mean.size()) {
            std::cerr << "ERROR: dim mismatch feat=" << feat.size()
                      << " zscore=" << zp.mean.size() << "\n";
            errores++;
            continue;
        }
        if (!aplicarZScore(feat, zp)) {
            errores++;
            continue;
        }

        // PCA
        auto red = aplicarPCAConModelo({ feat }, pca);
        if (red.empty() || red[0].empty()) {
            errores++;
            continue;
        }

        // Normalización L2 ELIMINADA (procesar_dataset.cpp NO la usa)
        // El SVM fue entrenado con vectores SIN normalización L2 post-PCA
        // normalizarVector(red[0]);  // ← COMENTADO: causaba scores incorrectos

        // Calcular TODOS los scores
        std::vector<double> scores = calcularTodosLosScores(red[0], svm);

        // Score genuino: score de la clase correcta
        double score_genuino = scores[idx_clase_real];
        scores_genuinos.push_back(score_genuino);

        // Scores impostores: scores de todas las demás clases
        for (size_t i = 0; i < scores.size(); ++i) {
            if ((int)i != idx_clase_real) {
                scores_impostores.push_back(scores[i]);
            }
        }

        num_muestras++;
        if (num_muestras % 10 == 0) {
            std::cout << "  Procesadas: " << num_muestras << " muestras\r" << std::flush;
        }
    }

    std::cout << "\n\nProcesamiento completo:\n";
    std::cout << "  - Muestras procesadas: " << num_muestras << "\n";
    std::cout << "  - Errores: " << errores << "\n";
    std::cout << "  - Comparaciones genuinas: " << scores_genuinos.size() << "\n";
    std::cout << "  - Comparaciones impostoras: " << scores_impostores.size() << "\n\n";

    if (scores_genuinos.empty() || scores_impostores.empty()) {
        std::cerr << "ERROR: No hay suficientes datos para calcular métricas\n";
        return 4;
    }

    // 3) Calcular estadísticas
    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    auto stddev = [&mean](const std::vector<double>& v) {
        double m = mean(v);
        double sq_sum = 0.0;
        for (double val : v) sq_sum += (val - m) * (val - m);
        return std::sqrt(sq_sum / v.size());
    };

    double mean_genuinos = mean(scores_genuinos);
    double std_genuinos = stddev(scores_genuinos);
    double mean_impostores = mean(scores_impostores);
    double std_impostores = stddev(scores_impostores);

    std::cout << "Estadísticas de scores:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Genuinos:   μ = " << mean_genuinos << ", σ = " << std_genuinos << "\n";
    std::cout << "  Impostores: μ = " << mean_impostores << ", σ = " << std_impostores << "\n";
    std::cout << "  Separación: " << (mean_genuinos - mean_impostores) << " (mayor es mejor)\n\n";

    // 4) Barrer umbrales y calcular FAR/FRR
    double min_score = std::min(
        *std::min_element(scores_genuinos.begin(), scores_genuinos.end()),
        *std::min_element(scores_impostores.begin(), scores_impostores.end())
    );
    double max_score = std::max(
        *std::max_element(scores_genuinos.begin(), scores_genuinos.end()),
        *std::max_element(scores_impostores.begin(), scores_impostores.end())
    );

    std::vector<MetricasVerificacion> metricas;
    const int num_umbrales = 1000;

    for (int i = 0; i <= num_umbrales; ++i) {
        double umbral = min_score + (max_score - min_score) * i / num_umbrales;

        // FAR: fracción de impostores aceptados (score >= umbral)
        int fa = std::count_if(scores_impostores.begin(), scores_impostores.end(),
                               [umbral](double s) { return s >= umbral; });
        double far = (double)fa / scores_impostores.size();

        // FRR: fracción de genuinos rechazados (score < umbral)
        int fr = std::count_if(scores_genuinos.begin(), scores_genuinos.end(),
                               [umbral](double s) { return s < umbral; });
        double frr = (double)fr / scores_genuinos.size();

        metricas.push_back({umbral, far, frr, std::abs(far - frr)});
    }

    // 5) Encontrar puntos de operación importantes

    // EER: Equal Error Rate (FAR ≈ FRR)
    auto it_eer = std::min_element(metricas.begin(), metricas.end(),
        [](const MetricasVerificacion& a, const MetricasVerificacion& b) {
            return a.err < b.err;
        });

    // FAR ≤ 1% (alta seguridad)
    auto it_far1 = std::find_if(metricas.rbegin(), metricas.rend(),
        [](const MetricasVerificacion& m) { return m.far <= 0.01; });

    // FAR ≤ 5% (seguridad media)
    auto it_far5 = std::find_if(metricas.rbegin(), metricas.rend(),
        [](const MetricasVerificacion& m) { return m.far <= 0.05; });

    // FAR ≤ 10% (usabilidad)
    auto it_far10 = std::find_if(metricas.rbegin(), metricas.rend(),
        [](const MetricasVerificacion& m) { return m.far <= 0.10; });

    // 6) Mostrar resultados
    std::cout << "==============================================\n";
    std::cout << "  UMBRALES RECOMENDADOS\n";
    std::cout << "==============================================\n\n";

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "1) UMBRAL EER (Balance FAR/FRR):\n";
    std::cout << "   Umbral: " << it_eer->umbral << "\n";
    std::cout << "   FAR:    " << (it_eer->far * 100) << "%\n";
    std::cout << "   FRR:    " << (it_eer->frr * 100) << "%\n";
    std::cout << "   → Uso: Sistema balanceado (defensa académica)\n\n";

    if (it_far1 != metricas.rend()) {
        std::cout << "2) UMBRAL ALTA SEGURIDAD (FAR ≤ 1%):\n";
        std::cout << "   Umbral: " << it_far1->umbral << "\n";
        std::cout << "   FAR:    " << (it_far1->far * 100) << "%\n";
        std::cout << "   FRR:    " << (it_far1->frr * 100) << "%\n";
        std::cout << "   → Uso: Acceso crítico (finanzas, datos sensibles)\n\n";
    }

    if (it_far5 != metricas.rend()) {
        std::cout << "3) UMBRAL SEGURIDAD MEDIA (FAR ≤ 5%):\n";
        std::cout << "   Umbral: " << it_far5->umbral << "\n";
        std::cout << "   FAR:    " << (it_far5->far * 100) << "%\n";
        std::cout << "   FRR:    " << (it_far5->frr * 100) << "%\n";
        std::cout << "   → Uso: Aplicaciones corporativas\n\n";
    }

    if (it_far10 != metricas.rend()) {
        std::cout << "4) UMBRAL ALTA USABILIDAD (FAR ≤ 10%):\n";
        std::cout << "   Umbral: " << it_far10->umbral << "\n";
        std::cout << "   FAR:    " << (it_far10->far * 100) << "%\n";
        std::cout << "   FRR:    " << (it_far10->frr * 100) << "%\n";
        std::cout << "   → Uso: Aplicaciones casuales, comodidad\n\n";
    }

    // 7) Guardar CSV detallado
    std::string csv_out = "out/metricas_verificacion_1_1.csv";
    std::ofstream f(csv_out);
    if (f.is_open()) {
        f << "umbral,far,frr,err\n";
        for (const auto& m : metricas) {
            f << m.umbral << "," << m.far << "," << m.frr << "," << m.err << "\n";
        }
        f.close();
        std::cout << "Métricas detalladas guardadas en: " << csv_out << "\n";
    }

    std::cout << "\n==============================================\n";
    std::cout << "  RECOMENDACIÓN PARA SERVIDOR.CPP:\n";
    std::cout << "==============================================\n\n";
    std::cout << "Actualizar línea 871 en servidor.cpp:\n\n";
    std::cout << "  // Antes:\n";
    std::cout << "  double UMBRAL_VERIFICACION = 0.25;  // arbitrario\n\n";
    std::cout << "  // Después:\n";
    std::cout << "  double UMBRAL_VERIFICACION = " << it_eer->umbral << ";  // EER optimizado\n\n";

    std::cout << "O usar variable de entorno para flexibilidad:\n";
    std::cout << "  double UMBRAL_VERIFICACION = getEnvDouble(\"UMBRAL_AUTENTICACION\", "
              << it_eer->umbral << ");\n\n";

    return 0;
}
