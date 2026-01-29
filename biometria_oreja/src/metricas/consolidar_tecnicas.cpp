#include "metricas/consolidar_tecnicas.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace fs = std::filesystem;

static bool leerLineaCSV6(const std::string& line, std::vector<std::string>& parts) {
    parts.clear();
    parts.reserve(6);

    size_t start = 0;
    while (true) {
        size_t pos = line.find(',', start);
        if (pos == std::string::npos) {
            parts.push_back(line.substr(start));
            break;
        }
        parts.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return parts.size() >= 6;
}

static void acumularArchivo(
    const std::string& path,
    std::unordered_map<std::string, double>& tMax,
    std::unordered_map<std::string, double>& cSum,
    std::unordered_map<std::string, size_t>& ramMax
) {
    std::ifstream f(path);
    if (!f.is_open()) return;

    std::string line;
    bool first = true;
    std::vector<std::string> parts;

    while (std::getline(f, line)) {
        if (line.empty()) continue;

        if (first && line.rfind("nombre,fase,", 0) == 0) { first = false; continue; }
        first = false;

        if (!leerLineaCSV6(line, parts)) continue;

        const std::string& fase = parts[1];

        double t = 0.0, c = 0.0;
        size_t rm = 0;

        try {
            t = std::stod(parts[2]);
            c = std::stod(parts[3]);
            rm = (size_t)std::stoull(parts[5]);
        } catch (...) {
            continue;
        }

        tMax[fase] = std::max(tMax[fase], t);
        cSum[fase] += c;
        ramMax[fase] = std::max(ramMax[fase], rm);
    }
}

void consolidarTecnicasParalelo(
    const std::string& carpetaWorkersCSV,
    const std::string& csvMainFases,
    const std::string& outCSV
) {
    std::unordered_map<std::string, double> tMax;
    std::unordered_map<std::string, double> cSum;
    std::unordered_map<std::string, size_t> ramMax;

    if (fs::exists(carpetaWorkersCSV)) {
        for (const auto& entry : fs::directory_iterator(carpetaWorkersCSV)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;
            acumularArchivo(entry.path().string(), tMax, cSum, ramMax);
        }
    }

    if (fs::exists(csvMainFases)) {
        acumularArchivo(csvMainFases, tMax, cSum, ramMax);
    }

    // Crear carpeta solo si hay parent_path
    const auto parent = fs::path(outCSV).parent_path();
    if (!parent.empty()) fs::create_directories(parent);

    std::ofstream out(outCSV, std::ios::out);
    if (!out.is_open()) return;

    out << "nombre,fase,tiempo_s,cpu_s,cpu_pct,ram_max_kb\n";

    std::vector<std::string> fases;
    fases.reserve(tMax.size());
    for (const auto& kv : tMax) fases.push_back(kv.first);
    std::sort(fases.begin(), fases.end());

    for (const auto& fase : fases) {
        double t = tMax[fase];
        double c = cSum[fase];
        double cpuPct = (t > 0.0) ? (100.0 * c / t) : 0.0;
        size_t rm = ramMax.count(fase) ? ramMax[fase] : 0;

        out << "Procesar_Dataset" << "," << fase << "," << t << "," << c << "," << cpuPct << "," << rm << "\n";
    }
}
