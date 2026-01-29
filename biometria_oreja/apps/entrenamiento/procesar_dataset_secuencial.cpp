// procesar_dataset_secuencial.cpp (docker-friendly: argv + env fallback)

#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/clahe.h"
#include "preprocesamiento/bilateral_filter.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "preprocesamiento/aumentar_dataset.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/dataset_loader.h"
#include "utilidades/guardar_csv.h"
#include "utilidades/pca_utils.h"
#include "utilidades/normalizacion.h"
#include "metricas/rendimiento.h"
#include "utilidades/svm_ova_utils.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <map>
#include <memory>
#include <algorithm>
#include <cstdlib> // getenv

/* namespace fs = std::filesystem;

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

static int resolverPCA(int argc, char** argv, int def = 70) {
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
// Progreso
// ==========================
static size_t imagenesProcesadas = 0;
static size_t totalImagenes = 0;

// ==========================
// Features
// ==========================
static std::vector<double> extraerFeaturesDesde128(const uint8_t* img128, const uint8_t* mask128) {
    return calcularLBPPorBloquesRobustoNorm(img128, mask128, 128, 128, 4, 4, 20, true);
}

static void escribirEncabezadoFasesCSV(const std::string& ruta) {
    fs::create_directories(fs::path(ruta).parent_path());
    std::ofstream f(ruta, std::ios::out);
    if (f.is_open()) {
        f << "nombre,fase,tiempo_s,cpu_s,cpu_pct,ram_max_kb\n";
    }
}

static void procesarImagenSecuencial(
    const std::string& ruta,
    int etiqueta,
    std::vector<std::vector<double>>& X,
    std::vector<int>& y,
    MedidorRendimiento& med
) {
    med.marcar("cargar_imagen");
    int ancho = 0, alto = 0, canales = 0;
    unsigned char* imgRGB = cargarImagen(ruta, ancho, alto, canales, 3);
    if (!imgRGB) {
        std::cerr << "\nError cargando imagen: " << ruta << "\n";
        return;
    }

    med.marcar("gris");
    auto gris = convertirAGris(imgRGB, ancho, alto);
    delete[] imgRGB;

    med.marcar("bilateral");
    auto bilateral = preprocesarImagenOreja(gris.get(), ancho, alto);

    med.marcar("iluminacion");
    auto iluminada = ajusteIluminacionBiometriaV2(bilateral.get(), ancho, alto);

    med.marcar("mascara");
    auto mascara = detectarRegionOreja(bilateral.get(), ancho, alto);

    med.marcar("clahe");
    auto conCLAHE = aplicarCLAHELocal(iluminada.get(), mascara.get(), ancho, alto, 12);

    med.marcar("recorte");
    int wRec = 0, hRec = 0;
    auto rec = recortarBoundingBox(conCLAHE.get(), mascara.get(), ancho, alto, wRec, hRec, 10);
    auto maskRec = recortarBoundingBoxMascara(mascara.get(), ancho, alto, wRec, hRec, 10);

    med.marcar("resize");
    auto img128 = redimensionarParaBiometria(rec.get(), wRec, hRec, 128, 128);
    auto mask128 = redimensionarMascaraSimple(maskRec.get(), wRec, hRec, 128, 128);
    dilatacion3x3_binaria(mask128.get(), 128, 128);

    if (X.capacity() < X.size() + 8) {
        X.reserve(X.size() + 8);
        y.reserve(y.size() + 8);
    }

    med.marcar("lbp_base");
    X.push_back(extraerFeaturesDesde128(img128.get(), mask128.get()));
    y.push_back(etiqueta);

    med.marcar("aumento");
    auto aumentadas128 = aumentarImagenFotometrica(img128.get(), 128, 128, ruta);

    med.marcar("lbp_aumentadas");
    for (const auto& [imgAum, _] : aumentadas128) {
        X.push_back(extraerFeaturesDesde128(imgAum.get(), mask128.get()));
        y.push_back(etiqueta);
    }

    ++imagenesProcesadas;
    if ((imagenesProcesadas % 25) == 0 || imagenesProcesadas == totalImagenes) {
        std::cout << "\rProgreso: " << imagenesProcesadas << " / " << totalImagenes << std::flush;
    }
}

static void ejecutarSecuencial(
    const std::vector<std::string>& rutas,
    const std::vector<int>& etiquetas,
    std::vector<std::vector<double>>& X,
    std::vector<int>& y,
    MedidorRendimiento& med
) {
    const size_t n = rutas.size();
    for (size_t i = 0; i < n; ++i) {
        procesarImagenSecuencial(rutas[i], etiquetas[i], X, y, med);
    }
} */

int main(int argc, char** argv) {
    /* const std::string outDir = resolverOutDir();
    asegurarDir(outDir);

    fs::remove(joinPath(outDir, "rendimiento_secuencial.csv"));
    fs::remove(joinPath(outDir, "rendimiento_fases_secuencial.csv"));

    MedidorRendimiento medidor("Procesar_Dataset_Secuencial");
    medidor.iniciar();
    medidor.marcar("cargar_rutas");

    const std::string rutaDataset = resolverRutaDataset(argc, argv);
    const int componentesPCA = resolverPCA(argc, argv, 70);

    std::cout << "Dataset: " << rutaDataset << "\n";
    std::cout << "PCA: " << componentesPCA << "\n";
    std::cout << "OUT_DIR: " << outDir << "\n";

    std::vector<std::string> rutas;
    std::vector<int> etiquetas;
    std::map<int, int> mapaRealAInterna;
    cargarRutasDataset(rutaDataset, rutas, etiquetas, mapaRealAInterna);

    totalImagenes = rutas.size();
    imagenesProcesadas = 0;

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    X.reserve(totalImagenes * 8);
    y.reserve(totalImagenes * 8);

    ejecutarSecuencial(rutas, etiquetas, X, y, medidor);

    std::cout << "\nMuestras: " << X.size() << "\n";

    medidor.marcar("pca_fit");
    ModeloPCA modeloPCA = entrenarPCA(X, componentesPCA);
    guardarModeloPCA(joinPath(outDir, "modelo_pca.dat"), modeloPCA);

    medidor.marcar("pca_transform");
    auto Xpca = aplicarPCAConModelo(X, modeloPCA);

    medidor.marcar("normalizacion");
    for (auto& v : Xpca) normalizarVector(v);

    medidor.marcar("guardar_csvs");
    guardarCSV(joinPath(outDir, "caracteristicas_fusionadas.csv"), X, y, ';');
    guardarCSV(joinPath(outDir, "caracteristicas_pca.csv"), Xpca, y, ';');

    medidor.marcar("fin");
    medidor.finalizar();
    medidor.imprimirResumen();

    medidor.guardarEnArchivo(joinPath(outDir, "rendimiento_secuencial.csv"));

    const std::string fases = joinPath(outDir, "rendimiento_fases_secuencial.csv");
    escribirEncabezadoFasesCSV(fases);
    medidor.guardarFasesCSV(fases);

    return 0; */
}