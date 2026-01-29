#include "cargar_imagen.h"
#include "preprocesamiento/convertir_a_gris.h"
#include "preprocesamiento/bilateral_filter.h"
#include "preprocesamiento/ajuste_iluminacion.h"
#include "preprocesamiento/redimensionar_imagen.h"
#include "preprocesamiento/mejoras_preprocesamiento.h"
#include "preprocesamiento/aumentar_dataset.h"
#include "extraccion_caracteristicas/lbp.h"
#include "utilidades/guardar_pgm.h"
#include "utilidades/svm_ova_utils.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <chrono>

namespace fs = std::filesystem;
using Reloj = std::chrono::steady_clock;

int main() {

/*     // Ruta fija de la imagen para pruebas
    std::string rutaImagen = "C:/Users/Usuario/Desktop/Universidad/Tesis_Biometria/Dataset/subset_50/000_down_ear.jpg";

    // Carpeta donde se guardarán las imágenes del pipeline
    std::string carpetaOut = "out/ejemplo_pipeline";
    fs::create_directories(carpetaOut);

    int ancho, alto, canales;
    unsigned char* imgRGB = cargarImagen(rutaImagen, ancho, alto, canales, 3);
    if (!imgRGB) {
        std::cerr << "Error cargando imagen: " << rutaImagen << "\n";
        return 1;
    }

    auto t0 = Reloj::now();

    // 1) Gris
    auto gris = convertirAGris(imgRGB, ancho, alto);
    guardarImagenPGM(carpetaOut + "/01_gris.pgm", gris.get(), ancho, alto);

    // 2) Filtro mediana + bilateral
    auto bilateral = preprocesarImagenOreja(gris.get(), ancho, alto);
    guardarImagenPGM(carpetaOut + "/02_bilateral.pgm", bilateral.get(), ancho, alto);

    // 3) Ajuste iluminación + máscara
    auto iluminada = ajusteIluminacionBiometriaV2(bilateral.get(), ancho, alto);
    auto mascara = detectarRegionOreja(bilateral.get(), ancho, alto);
    guardarImagenPGM(carpetaOut + "/03_iluminada.pgm", iluminada.get(), ancho, alto);
    guardarImagenPGM(carpetaOut + "/04_mascara_oreja.pgm", mascara.get(), ancho, alto);

    // 4) CLAHE local
    auto conCLAHE = aplicarCLAHELocal(iluminada.get(), mascara.get(), ancho, alto, 12);
    guardarImagenPGM(carpetaOut + "/05_clahe.pgm", conCLAHE.get(), ancho, alto);

    // 5) Bounding Box
    int wRec = 0, hRec = 0;
    auto recortada = recortarBoundingBox(conCLAHE.get(), mascara.get(), ancho, alto, wRec, hRec, 10);
    auto mascaraRec = recortarBoundingBoxMascara(mascara.get(), ancho, alto, wRec, hRec, 10);

    if (!recortada || !mascaraRec) {
        std::cerr << "No se pudo recortar la oreja\n";
        delete[] imgRGB;
        return 1;
    }

    guardarImagenPGM(carpetaOut + "/06_recorte_oreja.pgm", recortada.get(), wRec, hRec);
    guardarImagenPGM(carpetaOut + "/07_recorte_mascara.pgm", mascaraRec.get(), wRec, hRec);

    // 6) Redimensionar a 128x128
    auto redimensionada = redimensionarParaBiometria(recortada.get(), wRec, hRec, 128, 128);
    auto mascaraRedim = redimensionarMascaraSimple(mascaraRec.get(), wRec, hRec, 128, 128);
    dilatacion3x3_binaria(mascaraRedim.get(), 128, 128);

    guardarImagenPGM(carpetaOut + "/08_redimensionada.pgm", redimensionada.get(), 128, 128);
    guardarImagenPGM(carpetaOut + "/09_mascara_redim.pgm", mascaraRedim.get(), 128, 128);

    // 7) Aumentación fotométrica
    auto aumentadas128 = aumentarImagenFotometrica(redimensionada.get(), 128, 128, "oreja");
    fs::create_directories(carpetaOut + "/aumentadas");

    int idx = 0;
    for (auto& par : aumentadas128) {
        const auto& imgAum = par.first;
        guardarImagenPGM(
            carpetaOut + "/aumentadas/aug_" + std::to_string(idx++) + ".pgm",
            imgAum.get(),
            128, 128
        );
    }

    auto t1 = Reloj::now();
    double segundos = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;
    std::cout << "Tiempo total pipeline para 1 imagen: " << segundos << " s\n";

    delete[] imgRGB; */
    return 0;
}
