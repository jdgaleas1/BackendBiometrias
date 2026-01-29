#include "svm/cargar_csv.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace {

    std::vector<double> parseLinea(const std::string& linea, char delim) {
        std::stringstream ss(linea);
        std::string token;
        std::vector<double> valores;
        while (std::getline(ss, token, delim)) {
            if (!token.empty())
                valores.push_back(std::stod(token));
        }
        return valores;
    }

}

bool cargarCSV(const std::string& ruta,
    std::vector<std::vector<double>>& caracteristicas,
    std::vector<int>& etiquetas,
    char delimitador) {

    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "❌ Error al abrir archivo CSV: " << ruta << "\n";
        return false;
    }

    std::string linea;
    while (std::getline(archivo, linea)) {
        if (linea.empty()) continue;

        auto valores = parseLinea(linea, delimitador);
        if (valores.size() < 2) continue;

        etiquetas.push_back(static_cast<int>(valores.back()));
        valores.pop_back();
        caracteristicas.push_back(std::move(valores));
    }

    return true;
}

bool cargarCSVSinEtiquetas(const std::string& ruta,
    std::vector<std::vector<double>>& caracteristicas,
    char delimitador) {

    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "❌ Error al abrir archivo CSV: " << ruta << "\n";
        return false;
    }

    std::string linea;
    while (std::getline(archivo, linea)) {
        if (linea.empty()) continue;

        auto valores = parseLinea(linea, delimitador);
        if (!valores.empty())
            caracteristicas.push_back(std::move(valores));
    }

    return true;
}
