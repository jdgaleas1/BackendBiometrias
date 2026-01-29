// ============================================================================
// EVALUADOR DE MÉTRICAS POR TÉCNICA vs ACCURACY
// ============================================================================
// Propósito: Demostrar que las técnicas de preprocesamiento mejoran accuracy
// Analiza cada imagen del dataset:
//   1. Calcula métricas ANTES y DESPUÉS de cada técnica
//   2. Predice con el modelo entrenado
//   3. Correlaciona métricas con accuracy
//
// Salida: CSV con métricas + accuracy para análisis estadístico
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <memory>
#include <map>

#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/normalizacion.h"
#include "utilidades/pca_utils.h"
#include "utilidades/svm_ova_utils.h"
#include "utilidades/zscore_params.h"
#include "svm/svm_prediccion.h"

namespace fs = std::filesystem;

// ============================================================================
// MÉTRICAS DE CALIDAD DE IMAGEN
// ============================================================================

struct ImageMetrics {
    double mean = 0.0;
    double stddev = 0.0;
    int min_val = 255;
    int max_val = 0;
    double entropy = 0.0;
    double michelson_contrast = 0.0;
    double rms_contrast = 0.0;
    double dynamic_range = 0.0;
};

// Entropía de Shannon (Shannon 1948)
double calcEntropy(const uint8_t* img, int w, int h) {
    if (!img || w <= 0 || h <= 0) return 0.0;
    
    std::vector<int> hist(256, 0);
    const int N = w * h;
    
    for (int i = 0; i < N; ++i) hist[img[i]]++;
    
    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            double p = (double)hist[i] / N;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// Calcular todas las métricas
ImageMetrics calcMetrics(const uint8_t* img, int w, int h) {
    ImageMetrics m;
    const int N = w * h;
    if (!img || N <= 0) return m;

    long long sum = 0, sum2 = 0;
    
    for (int i = 0; i < N; ++i) {
        int v = (int)img[i];
        sum += v;
        sum2 += 1LL * v * v;
        if (v < m.min_val) m.min_val = v;
        if (v > m.max_val) m.max_val = v;
    }

    m.mean = (double)sum / (double)N;
    double var = (double)sum2 / (double)N - m.mean * m.mean;
    m.stddev = std::sqrt(var > 0 ? var : 0);
    
    m.entropy = calcEntropy(img, w, h);
    
    // Michelson Contrast (Michelson 1927)
    if (m.max_val + m.min_val > 0) {
        m.michelson_contrast = (double)(m.max_val - m.min_val) / (double)(m.max_val + m.min_val);
    }
    
    m.rms_contrast = m.stddev; // RMS = stddev (Peli 1990)
    m.dynamic_range = m.max_val - m.min_val;

    return m;
}

// PSNR entre dos imágenes (Wang 2004)
double calcPSNR(const uint8_t* img1, const uint8_t* img2, int w, int h) {
    if (!img1 || !img2 || w <= 0 || h <= 0) return 0.0;
    
    const int N = w * h;
    double mse = 0.0;
    
    for (int i = 0; i < N; ++i) {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }
    mse /= N;
    
    if (mse < 1e-10) return 100.0; // Imágenes idénticas
    
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// ============================================================================
// PIPELINE CON MÉTRICAS
// ============================================================================

struct PipelineMetrics {
    // Métricas por fase
    ImageMetrics original;
    ImageMetrics resize;
    ImageMetrics clahe;
    ImageMetrics bilateral;
    
    // Métricas de cambio
    double psnr_clahe = 0.0;        // CLAHE vs Resize
    double psnr_bilateral = 0.0;     // Bilateral vs CLAHE
    
    // Predicción
    int clase_predicha = -1;
    int clase_real = -1;
    bool prediccion_correcta = false;
    double score_top1 = 0.0;
    double margen = 0.0;
};

PipelineMetrics procesarImagenCompleta(
    const std::string& rutaImg,
    int claseReal,
    const ModeloPCA& modeloPCA,
    const ModeloSVM& modeloSVM,
    const ZScoreParams& zscore
) {
    PipelineMetrics pm;
    pm.clase_real = claseReal;
    
    // 1. Cargar
    int w, h, c;
    auto rgb = cargarImagen(rutaImg.c_str(), w, h, c, 3);
    if (!rgb) return pm;
    
    auto gris = convertirAGris(rgb, w, h);
    delete[] rgb;
    
    pm.original = calcMetrics(gris.get(), w, h);
    
    // 2. Resize
    auto img128 = redimensionarParaBiometria(gris.get(), w, h, 128, 128);
    pm.resize = calcMetrics(img128.get(), 128, 128);
    
    // 3. CLAHE
    auto img_clahe = aplicarCLAHE(img128.get(), 128, 128, 8, 8, 2.0);
    pm.clahe = calcMetrics(img_clahe.get(), 128, 128);
    pm.psnr_clahe = calcPSNR(img128.get(), img_clahe.get(), 128, 128);
    
    // 4. Bilateral
    auto img_bilateral = aplicarBilateral(img_clahe.get(), 128, 128, 3.0, 50.0);
    pm.bilateral = calcMetrics(img_bilateral.get(), 128, 128);
    pm.psnr_bilateral = calcPSNR(img_clahe.get(), img_bilateral.get(), 128, 128);
    
    // 5. LBP Multi-Scale
    auto mask = crearMascaraElipticaFija(128, 128);
    auto features = calcularLBPMultiEscalaPorBloquesRobustoNorm(
        img_bilateral.get(), mask.get(), 128, 128, 6, 6, 200, true
    );
    
    if (features.empty()) return pm;
    
    // 6. Z-score
    if (features.size() != zscore.mean.size()) return pm;
    if (!aplicarZScore(features, zscore)) return pm;
    
    // 7. PCA
    auto reducidas = aplicarPCAConModelo({features}, modeloPCA);
    if (reducidas.empty() || reducidas[0].empty()) return pm;
    
    // 8. Predicción
    double s1, s2;
    int pred;
    if (predictOVAScore(modeloSVM, reducidas[0], s1, s2, pred) >= 0) {
        pm.clase_predicha = pred;
        pm.score_top1 = s1;
        pm.margen = s1 - s2;
        pm.prediccion_correcta = (pred == claseReal);
    }
    
    return pm;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    const std::string DATASET = "../dataset/";
    const std::string OUT_CSV = "../out/analisis_metricas_tecnicas.csv";
    
    std::cout << "Cargando modelos..." << std::endl;
    
    // Cargar Z-score
    ZScoreParams zscore;
    if (!cargarZScoreParams("../out/zscore_params.dat", zscore, ';')) {
        std::cerr << "ERROR: No se pudo cargar zscore_params.dat" << std::endl;
        return 1;
    }
    
    // Cargar PCA
    ModeloPCA pca = cargarModeloPCA("../out/modelo_pca.dat");
    if (pca.componentes.empty()) {
        std::cerr << "ERROR: No se pudo cargar modelo PCA" << std::endl;
        return 1;
    }
    
    // Cargar SVM
    ModeloSVM svm;
    if (!cargarModeloSVM("../out/modelo_svm.svm", svm)) {
        std::cerr << "ERROR: No se pudo cargar modelo SVM" << std::endl;
        return 1;
    }
    
    std::cout << "Modelos cargados: " << svm.clases.size() << " clases" << std::endl;
    
    // Leer mapa de etiquetas
    std::map<int, std::string> mapaEtiquetas;
    std::ifstream fmapa("../out/mapa_etiquetas.txt");
    if (fmapa.is_open()) {
        std::string line;
        while (std::getline(fmapa, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                int clase = std::stoi(line.substr(0, pos));
                mapaEtiquetas[clase] = line.substr(pos+1);
            }
        }
    }
    
    // Procesar dataset
    std::vector<fs::path> imagenes;
    for (const auto& entry : fs::directory_iterator(DATASET)) {
        if (entry.path().extension() == ".jpg") {
            imagenes.push_back(entry.path());
        }
    }
    
    std::sort(imagenes.begin(), imagenes.end());
    
    std::cout << "Procesando " << imagenes.size() << " imágenes..." << std::endl;
    
    // Abrir CSV
    std::ofstream csv(OUT_CSV);
    csv << "imagen,clase_real,clase_pred,correcto,score,margen,"
        << "mean_orig,std_orig,entropy_orig,mc_orig,"
        << "mean_resize,std_resize,entropy_resize,mc_resize,"
        << "mean_clahe,std_clahe,entropy_clahe,mc_clahe,rms_clahe,psnr_clahe,"
        << "mean_bilateral,std_bilateral,entropy_bilateral,mc_bilateral,rms_bilateral,psnr_bilateral"
        << std::endl;
    
    int procesadas = 0;
    for (const auto& path : imagenes) {
        std::string nombre = path.filename().string();
        
        // Extraer clase del nombre (ej: 001_front.jpg -> clase 1)
        int clase = -1;
        try {
            std::string prefijo = nombre.substr(0, 3);
            clase = std::stoi(prefijo);
        } catch (...) {
            continue;
        }
        
        auto pm = procesarImagenCompleta(path.string(), clase, pca, svm, zscore);
        
        csv << nombre << ","
            << pm.clase_real << ","
            << pm.clase_predicha << ","
            << (pm.prediccion_correcta ? 1 : 0) << ","
            << pm.score_top1 << ","
            << pm.margen << ","
            << pm.original.mean << "," << pm.original.stddev << "," << pm.original.entropy << "," << pm.original.michelson_contrast << ","
            << pm.resize.mean << "," << pm.resize.stddev << "," << pm.resize.entropy << "," << pm.resize.michelson_contrast << ","
            << pm.clahe.mean << "," << pm.clahe.stddev << "," << pm.clahe.entropy << "," << pm.clahe.michelson_contrast << "," << pm.clahe.rms_contrast << "," << pm.psnr_clahe << ","
            << pm.bilateral.mean << "," << pm.bilateral.stddev << "," << pm.bilateral.entropy << "," << pm.bilateral.michelson_contrast << "," << pm.bilateral.rms_contrast << "," << pm.psnr_bilateral
            << std::endl;
        
        procesadas++;
        if (procesadas % 100 == 0) {
            std::cout << "Procesadas: " << procesadas << " / " << imagenes.size() << std::endl;
        }
    }
    
    csv.close();
    std::cout << "\n✓ Análisis completo guardado en: " << OUT_CSV << std::endl;
    std::cout << "Total procesadas: " << procesadas << " imágenes" << std::endl;
    
    return 0;
}
