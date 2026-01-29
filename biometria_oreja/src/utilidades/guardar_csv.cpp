#include "utilidades/guardar_csv.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

bool guardarCSV(const std::string& rutaArchivo,
                const std::vector<std::vector<double>>& datos,
                const std::vector<int>& etiquetas,
                char delimitador)
{
    if (datos.empty() || etiquetas.empty()) {
        std::cerr << "Datos vacios, no se genera el CSV.\n";
        return false;
    }

    if (datos.size() != etiquetas.size()) {
        std::cerr << "Error: el numero de muestras y etiquetas no coincide.\n";
        return false;
    }

    // Crear carpeta solo si hay parent_path (evita fallo si rutaArchivo no tiene carpeta)
    const auto parent = fs::path(rutaArchivo).parent_path();
    if (!parent.empty()) fs::create_directories(parent);

    std::ofstream archivo(rutaArchivo);
    if (!archivo.is_open()) {
        std::cerr << "❌ No se pudo abrir el archivo para guardar: " << rutaArchivo << "\n";
        return false;
    }

    for (size_t i = 0; i < datos.size(); ++i) {
        const auto& fila = datos[i];

        for (size_t j = 0; j < fila.size(); ++j) {
            if (j) archivo << delimitador;
            archivo << fila[j];
        }
        archivo << delimitador << etiquetas[i] << "\n";
    }

    return true;
}
