#include "dataset.h"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

// ============================================================================
// I/O DE DATASETS BINARIOS
// ============================================================================

bool cargarDatasetBinario(const std::string& ruta,
    std::vector<std::vector<AudioSample>>& X,
    std::vector<int>& y) {

    std::ifstream in(ruta, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << std::endl;
        return false;
    }

    std::cout << "-> Cargando caracteristicas: " << ruta << std::endl;

    X.clear();
    y.clear();

    int muestras_cargadas = 0;
    std::set<int> clases_unicas;

    // Formato por muestra: [dim:int][features:AudioSample*dim][label:int]
    while (in.peek() != EOF) {
        int dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));

        if (in.eof()) break;

        if (dim <= 0 || dim > 10000) {
            std::cerr << "! Error: Dimension invalida: " << dim << std::endl;
            in.close();
            return false;
        }

        std::vector<AudioSample> features(dim);
        in.read(reinterpret_cast<char*>(features.data()), sizeof(AudioSample) * dim);

        int label;
        in.read(reinterpret_cast<char*>(&label), sizeof(int));

        if (in.fail()) {
            std::cerr << "! Error de lectura en muestra " << muestras_cargadas << std::endl;
            break;
        }

        X.push_back(features);
        y.push_back(label);
        clases_unicas.insert(label);
        muestras_cargadas++;
    }

    in.close();

    if (muestras_cargadas == 0) {
        std::cerr << "! Error: No se cargaron muestras del archivo" << std::endl;
        return false;
    }

    std::cout << "   & Dataset cargado:" << std::endl;
    std::cout << "      Muestras: " << muestras_cargadas << std::endl;
    std::cout << "      Dimension: " << (X.empty() ? 0 : X[0].size()) << std::endl;
    std::cout << "      Clases: " << clases_unicas.size() << std::endl;

    return true;
}

Dataset cargarDatasetBinario(const std::string& ruta) {
    Dataset dataset;

    if (!cargarDatasetBinario(ruta, dataset.X, dataset.y)) {
        std::cerr << "! Error al cargar dataset" << std::endl;
        return Dataset();
    }

    return dataset;
}

bool guardarDatasetBinario(const std::string& ruta,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y) {

    if (X.size() != y.size()) {
        std::cerr << "! Error: X e y tienen tamanos diferentes" << std::endl;
        return false;
    }

    if (X.empty()) {
        std::cerr << "! Error: Dataset vacio" << std::endl;
        return false;
    }

    std::ofstream out(ruta, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta << std::endl;
        return false;
    }

    std::cout << "-> Guardando dataset binario: " << ruta << std::endl;

    int dim = static_cast<int>(X[0].size());

    // Formato por muestra: [dim:int][features:AudioSample*dim][label:int]
    for (size_t i = 0; i < X.size(); ++i) {
        if (static_cast<int>(X[i].size()) != dim) {
            std::cerr << "! Error: Dimension inconsistente en muestra " << i << std::endl;
            out.close();
            return false;
        }

        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(X[i].data()), sizeof(AudioSample) * dim);
        out.write(reinterpret_cast<const char*>(&y[i]), sizeof(int));
    }

    out.close();

    std::cout << "   & Dataset guardado:" << std::endl;
    std::cout << "      Muestras: " << X.size() << std::endl;
    std::cout << "      Dimension: " << dim << std::endl;

    return true;
}

bool guardarDatasetBinario(const std::string& ruta, const Dataset& dataset) {
    return guardarDatasetBinario(ruta, dataset.X, dataset.y);
}

/**
 * Agrega nuevas muestras a un dataset binario existente (INCREMENTAL)
 * Solo escribe al final del archivo, sin recargar todo
 */
bool agregarMuestrasDataset(const std::string& ruta,
    const std::vector<std::vector<AudioSample>>& nuevas_X,
    const std::vector<int>& nuevas_y) {

    if (nuevas_X.size() != nuevas_y.size()) {
        std::cerr << "! Error: nuevas_X e nuevas_y tienen tama�os diferentes" << std::endl;
        return false;
    }

    if (nuevas_X.empty()) {
        std::cerr << "! Error: No hay nuevas muestras para agregar" << std::endl;
        return false;
    }

    // Abrir en modo append (agregar al final)
    std::ofstream out(ruta, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << " en modo append" << std::endl;
        return false;
    }

    std::cout << "-> Agregando muestras al dataset: " << ruta << std::endl;

    int dim = static_cast<int>(nuevas_X[0].size());

    // Formato por muestra: [dim:int][features:AudioSample*dim][label:int]
    for (size_t i = 0; i < nuevas_X.size(); ++i) {
        if (static_cast<int>(nuevas_X[i].size()) != dim) {
            std::cerr << "! Error: Dimensi�n inconsistente en muestra " << i << std::endl;
            out.close();
            return false;
        }

        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(nuevas_X[i].data()), sizeof(AudioSample) * dim);
        out.write(reinterpret_cast<const char*>(&nuevas_y[i]), sizeof(int));
    }

    out.close();

    std::cout << "   & Muestras agregadas: " << nuevas_X.size() 
              << " (dim=" << dim << ")" << std::endl;

    return true;
}

// ============================================================================
// I/O DE MAPEO DE SPEAKERS
// ============================================================================

bool guardarMapeoSpeakers(const std::string& ruta, const SpeakerMapping& mapping) {
    std::ofstream out(ruta);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta << std::endl;
        return false;
    }

    std::cout << "-> Guardando mapeo de speakers: " << ruta << std::endl;

    // Header
    out << "speaker_id,index\n";

    // Mapeo ordenado por �ndice
    for (const auto& [idx, speaker_id] : mapping.index_to_speaker) {
        out << speaker_id << "," << idx << "\n";
    }

    out.close();

    std::cout << "   & Mapeo guardado: " << mapping.size() << " speakers" << std::endl;

    return true;
}

SpeakerMapping cargarMapeoSpeakers(const std::string& ruta) {
    SpeakerMapping mapping;

    std::ifstream in(ruta);
    if (!in.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << std::endl;
        return mapping;
    }

    std::cout << "-> Cargando mapeo de speakers: " << ruta << std::endl;

    std::string line;

    // Saltar header
    if (!std::getline(in, line)) {
        std::cerr << "! Error: Archivo vacio" << std::endl;
        in.close();
        return mapping;
    }

    // Leer mapeos
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string speaker_str, index_str;

        if (!std::getline(iss, speaker_str, ',') || !std::getline(iss, index_str)) {
            std::cerr << "% Warning: Linea invalida: " << line << std::endl;
            continue;
        }

        try {
            int speaker_id = std::stoi(speaker_str);
            int index = std::stoi(index_str);

            mapping.speaker_to_index[speaker_id] = index;
            mapping.index_to_speaker[index] = speaker_id;
        }
        catch (const std::exception& e) {
            std::cerr << "% Warning: Error al parsear linea: " << line << std::endl;
        }
    }

    in.close();

    std::cout << "   & Mapeo cargado: " << mapping.size() << " speakers" << std::endl;

    return mapping;
}
