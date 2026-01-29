#include "utilidades/guardar_pgm.h"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

bool guardarImagenPGM(const std::string& ruta, const uint8_t* imagen, int ancho, int alto) {
    if (!imagen || ancho <= 0 || alto <= 0) {
        std::cerr << "Imagen invalida. No se puede guardar.\n";
        return false;
    }

    fs::create_directories(fs::path(ruta).parent_path());

    std::ofstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escribir: " << ruta << "\n";
        return false;
    }

    archivo << "P2\n" << ancho << " " << alto << "\n255\n";

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            archivo << static_cast<int>(imagen[y * ancho + x]) << " ";
        }
        archivo << "\n";
    }

    archivo.close();
    std::cout << "Imagen PGM guardada: " << ruta << "\n";
    return true;
}
