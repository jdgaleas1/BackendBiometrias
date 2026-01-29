#include "utilidades/dataset_loader.h"
#include <filesystem>
#include <regex>
#include <set>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

void cargarRutasDataset(const std::string& carpetaBase,
    std::vector<std::string>& rutas,
    std::vector<int>& etiquetas,
    std::map<int, int>& mapaEtiquetaRealAInterna)
{
    rutas.clear();
    etiquetas.clear();
    mapaEtiquetaRealAInterna.clear();

    const std::regex patron(R"((\d{3})_.*\.jpg)");
    std::set<int> etiquetasUnicas;
    std::vector<std::pair<std::string, int>> archivosTemporales;

    for (const auto& archivo : fs::recursive_directory_iterator(carpetaBase)) {
        if (archivo.is_regular_file()
            && archivo.path().extension() == ".jpg"
            ) {
            const std::string nombre = archivo.path().filename().string();
            std::smatch match;
            if (std::regex_match(nombre, match, patron)) {
                int etiquetaReal = std::stoi(match[1]);
                archivosTemporales.emplace_back(archivo.path().string(), etiquetaReal);
                etiquetasUnicas.insert(etiquetaReal);
            }
        }
    }

    // Mapear etiquetas reales a internas
    int claseInterna = 0;
    for (int real : etiquetasUnicas)
        mapaEtiquetaRealAInterna[real] = claseInterna++;

    // Guardar mapeo
    const char* outEnv = std::getenv("OUT_DIR");
    std::string outDir = (outEnv && *outEnv) ? std::string(outEnv) : "out";

    fs::create_directories(outDir);
    std::ofstream out((fs::path(outDir) / "mapa_etiquetas.txt").string());

    // Reservas (opcional, no cambia resultado)
    rutas.reserve(archivosTemporales.size());
    etiquetas.reserve(archivosTemporales.size());

    // Asignar rutas y etiquetas internas
    for (const auto& [ruta, real] : archivosTemporales) {
        rutas.push_back(ruta);
        etiquetas.push_back(mapaEtiquetaRealAInterna[real]);
    }

    std::cout << "Se encontraron " << etiquetasUnicas.size()
        << " clases unicas. Mapeo generado en out/mapa_etiquetas.txt\n";
}
