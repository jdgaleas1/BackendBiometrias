#include "svm.h"
#include "../../external/json.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// I/O DE MODELOS SVM
// ============================================================================
// Este archivo contiene SOLO funciones de entrada/salida de modelos SVM.
// Las funciones de I/O de datasets est�n en dataset_io.cpp del m�dulo
// process_dataset.
// ============================================================================

/**
 * Guarda un modelo SVM entrenado en formato binario
 *
 * Formato del archivo:
 * [numClases:size_t][dimensionCaracteristicas:int]
 * Para cada clase:
 *   [clase:int][pesos:AudioSample*dim][bias:AudioSample]
 *
 * @param ruta Ruta del archivo de salida
 * @param modelo Modelo SVM a guardar
 * @return true si se guard� exitosamente
 */
bool guardarModeloSVM(const std::string& ruta, const ModeloSVM& modelo) {
    std::ofstream out(ruta, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta << std::endl;
        return false;
    }

    std::cout << "-> Guardando modelo SVM: " << ruta << std::endl;

    // Escribir metadatos
    size_t numClases = modelo.clases.size();
    out.write(reinterpret_cast<const char*>(&numClases), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&modelo.dimensionCaracteristicas), sizeof(int));

    // Escribir cada clasificador binario
    for (size_t i = 0; i < numClases; ++i) {
        int clase = modelo.clases[i];
        out.write(reinterpret_cast<const char*>(&clase), sizeof(int));

        out.write(reinterpret_cast<const char*>(modelo.pesosPorClase[i].data()),
            sizeof(AudioSample) * modelo.dimensionCaracteristicas);

        out.write(reinterpret_cast<const char*>(&modelo.biasPorClase[i]), sizeof(AudioSample));
    }

    out.close();

    std::cout << "   & Modelo guardado: " << numClases << " clases, "
        << modelo.dimensionCaracteristicas << " caracteristicas" << std::endl;

    // ? CORRECCI�N 5: Logging de diagn�stico mejorado
    std::cout << "\n   @ Diagnostico del modelo guardado:" << std::endl;

    // Encontrar biases extremos
    AudioSample minBias = modelo.biasPorClase[0];
    AudioSample maxBias = modelo.biasPorClase[0];
    int claseMinBias = modelo.clases[0];
    int claseMaxBias = modelo.clases[0];

    for (size_t i = 1; i < modelo.clases.size(); ++i) {
        if (modelo.biasPorClase[i] < minBias) {
            minBias = modelo.biasPorClase[i];
            claseMinBias = modelo.clases[i];
        }
        if (modelo.biasPorClase[i] > maxBias) {
            maxBias = modelo.biasPorClase[i];
            claseMaxBias = modelo.clases[i];
        }
    }

    std::cout << "      Bias rango: [" << minBias << ", " << maxBias << "]" << std::endl;
    std::cout << "      Clase con mayor bias: " << claseMaxBias
        << " (bias=" << maxBias << ")" << std::endl;
    std::cout << "      Clase con menor bias: " << claseMinBias
        << " (bias=" << minBias << ")" << std::endl;

    // Calcular normas promedio de pesos
    AudioSample normaPromedio = 0.0;
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        AudioSample norma = 0.0;
        for (int j = 0; j < modelo.dimensionCaracteristicas; ++j) {
            norma += modelo.pesosPorClase[i][j] * modelo.pesosPorClase[i][j];
        }
        normaPromedio += sqrt(norma);
    }
    normaPromedio /= numClases;

    std::cout << "      ||w|| promedio: " << normaPromedio << std::endl;

    return true;
}

/**
 * Carga un modelo SVM desde formato binario
 *
 * @param ruta Ruta del archivo con el modelo
 * @return Modelo SVM cargado (vac�o si hay error)
 */
ModeloSVM cargarModeloSVM(const std::string& ruta) {
    ModeloSVM modelo;

    std::ifstream in(ruta, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << std::endl;
        return modelo;
    }

    std::cout << "-> Cargando modelo SVM: " << ruta << std::endl;

    // Leer metadatos
    size_t numClases;
    in.read(reinterpret_cast<char*>(&numClases), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&modelo.dimensionCaracteristicas), sizeof(int));

    if (numClases == 0 || modelo.dimensionCaracteristicas == 0) {
        std::cerr << "! Error: Modelo invalido o corrupto" << std::endl;
        in.close();
        return modelo;
    }

    // Validar l�mites razonables
    if (numClases > 1000 || modelo.dimensionCaracteristicas > 10000) {
        std::cerr << "! Error: Dimensiones del modelo fuera de rango razonable" << std::endl;
        in.close();
        return modelo;
    }

    // Reservar espacio
    modelo.clases.resize(numClases);
    modelo.pesosPorClase.resize(numClases);
    modelo.biasPorClase.resize(numClases);

    // Leer cada clasificador binario
    for (size_t i = 0; i < numClases; ++i) {
        int clase;
        in.read(reinterpret_cast<char*>(&clase), sizeof(int));
        modelo.clases[i] = clase;

        modelo.pesosPorClase[i].resize(modelo.dimensionCaracteristicas);
        in.read(reinterpret_cast<char*>(modelo.pesosPorClase[i].data()),
            sizeof(AudioSample) * modelo.dimensionCaracteristicas);

        in.read(reinterpret_cast<char*>(&modelo.biasPorClase[i]), sizeof(AudioSample));

        if (in.fail()) {
            std::cerr << "! Error: Fallo al leer clasificador de clase " << i << std::endl;
            in.close();
            return ModeloSVM(); // Retornar modelo vac�o
        }
    }

    in.close();

    std::cout << "   & Modelo cargado: " << numClases << " clases, "
        << modelo.dimensionCaracteristicas << " caracteristicas" << std::endl;

    return modelo;
}

// ============================================================================
// I/O MODULAR - CLASIFICADORES INDEPENDIENTES
// ============================================================================

/**
 * Guarda un clasificador binario individual en formato binario
 *
 * Formato del archivo class_XXXX.bin:
 * [dimension:int][pesos:AudioSample*dim][bias:AudioSample]
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param clase ID del usuario/clase
 * @param pesos Vector de pesos del clasificador
 * @param bias Bias del clasificador
 * @param dimension Dimensi�n de caracter�sticas
 * @return true si se guard� exitosamente
 */
bool guardarClasificadorBinario(
    const std::string& ruta_base,
    int clase,
    const ClasificadorBinario& clasificador
) {
    // Crear directorio si no existe
    fs::create_directories(ruta_base);

    std::string ruta = ruta_base + "class_" + std::to_string(clase) + ".bin";
    std::ofstream out(ruta, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta << std::endl;
        return false;
    }

    // Formato binario: [dimension][pesos][bias][plattA][plattB][thresholdOptimo]
    int dimension = static_cast<int>(clasificador.pesos.size());

    // 1. Dimension
    out.write(reinterpret_cast<const char*>(&dimension), sizeof(int));

    // 2. Pesos
    out.write(reinterpret_cast<const char*>(clasificador.pesos.data()),
        sizeof(AudioSample) * dimension);

    // 3. Bias
    out.write(reinterpret_cast<const char*>(&clasificador.bias), sizeof(AudioSample));

    // 4. Platt A y B (para calibracion de probabilidades)
    out.write(reinterpret_cast<const char*>(&clasificador.plattA), sizeof(AudioSample));
    out.write(reinterpret_cast<const char*>(&clasificador.plattB), sizeof(AudioSample));

    // 5. Threshold optimo
    out.write(reinterpret_cast<const char*>(&clasificador.thresholdOptimo), sizeof(AudioSample));

    out.close();

    std::cout << "   & Clasificador guardado: class_" << clase << ".bin "
        << "(dim=" << dimension << ", bias=" << clasificador.bias << ")" << std::endl;

    return true;
}

/**
 * Carga un clasificador binario individual desde archivo
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param clase ID del usuario/clase a cargar
 * @param clasificador Clasificador binario (salida)
 * @return true si se carg� exitosamente
 */
bool cargarClasificadorBinario(
    const std::string& ruta_base,
    int clase,
    ClasificadorBinario& clasificador
) {
    std::string ruta = ruta_base + "class_" + std::to_string(clase) + ".bin";
    std::ifstream in(ruta, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << std::endl;
        return false;
    }

    // Formato binario: [dimension][pesos][bias][plattA][plattB][thresholdOptimo]

    // 1. Dimension
    int dimension;
    in.read(reinterpret_cast<char*>(&dimension), sizeof(int));

    if (dimension <= 0 || dimension > 10000) {
        std::cerr << "! Error: Dimension invalida en clasificador: " << dimension << std::endl;
        in.close();
        return false;
    }

    // 2. Pesos
    clasificador.pesos.resize(dimension);
    in.read(reinterpret_cast<char*>(clasificador.pesos.data()),
        sizeof(AudioSample) * dimension);

    // 3. Bias
    in.read(reinterpret_cast<char*>(&clasificador.bias), sizeof(AudioSample));

    // 4. Platt A y B (si existen en el archivo)
    if (in.peek() != EOF) {
        in.read(reinterpret_cast<char*>(&clasificador.plattA), sizeof(AudioSample));
        in.read(reinterpret_cast<char*>(&clasificador.plattB), sizeof(AudioSample));

        // 5. Threshold optimo (si existe)
        if (in.peek() != EOF) {
            in.read(reinterpret_cast<char*>(&clasificador.thresholdOptimo), sizeof(AudioSample));
        }
    }

    if (in.fail()) {
        std::cerr << "! Error: Fallo al leer clasificador de clase " << clase << std::endl;
        in.close();
        return false;
    }

    in.close();
    return true;
}

/**
 * Guarda metadata del modelo en formato JSON
 *
 * Contenido de metadata.json:
 * {
 *   "num_classes": 42,
 *   "dimension": ?,
 *   "classes": [1001, 1002, ...]
 * }
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param num_clases N�mero total de clases
 * @param dimension Dimensi�n de caracter�sticas
 * @param clases Vector con IDs de todas las clases
 * @return true si se guard� exitosamente
 */
bool guardarMetadata(
    const std::string& ruta_base,
    int num_clases,
    int dimension,
    const std::vector<int>& clases
) {
    fs::create_directories(ruta_base);

    json j;
    j["num_classes"] = num_clases;
    j["dimension"] = dimension;
    j["classes"] = clases;

    std::string ruta = ruta_base + "metadata.json";
    std::ofstream out(ruta);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << ruta << std::endl;
        return false;
    }

    out << j.dump(4);  // Pretty print con indentaci�n de 4 espacios
    out.close();

    std::cout << "   & Metadata guardada: " << num_clases << " clases, "
        << dimension << " dimensiones" << std::endl;

    return true;
}

/**
 * Carga metadata del modelo desde JSON
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param num_clases N�mero total de clases (salida)
 * @param dimension Dimensi�n de caracter�sticas (salida)
 * @param clases Vector con IDs de todas las clases (salida)
 * @return true si se carg� exitosamente
 */
bool cargarMetadata(
    const std::string& ruta_base,
    int& num_clases,
    int& dimension,
    std::vector<int>& clases
) {
    std::string ruta = ruta_base + "metadata.json";
    std::ifstream in(ruta);
    if (!in.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << ruta << std::endl;
        return false;
    }

    json j;
    try {
        in >> j;
        num_clases = j["num_classes"];
        dimension = j["dimension"];
        clases = j["classes"].get<std::vector<int>>();
    }
    catch (const std::exception& e) {
        std::cerr << "! Error: JSON inv�lido: " << e.what() << std::endl;
        in.close();
        return false;
    }

    in.close();
    return true;
}

/**
 * Guarda un modelo SVM completo en formato modular
 * Crea un archivo .bin por cada clasificador + metadata.json
 *
 * Estructura generada:
 * model/
 *   +-- metadata.json
 *   +-- class_1001.bin
 *   +-- class_1002.bin
 *   +-- ...
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @param modelo Modelo SVM a guardar
 * @return true si se guard� exitosamente
 */
bool guardarModeloModular(const std::string& ruta_base, const ModeloSVM& modelo) {
    std::cout << "\n-> Guardando modelo SVM modular: " << ruta_base << std::endl;

    // Validar modelo
    if (modelo.clases.empty()) {
        std::cerr << "! Error: Modelo vac�o" << std::endl;
        return false;
    }

    // Guardar metadata
    if (!guardarMetadata(ruta_base, modelo.clases.size(),
        modelo.dimensionCaracteristicas, modelo.clases)) {
        return false;
    }

    // Guardar cada clasificador binario
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        ClasificadorBinario clasificador;
        clasificador.pesos = modelo.pesosPorClase[i];
        clasificador.bias = modelo.biasPorClase[i];
        clasificador.plattA = modelo.plattAPorClase[i];
        clasificador.plattB = modelo.plattBPorClase[i];
        clasificador.thresholdOptimo = modelo.thresholdsPorClase[i];

        if (!guardarClasificadorBinario(ruta_base, modelo.clases[i], clasificador)) {
            std::cerr << "! Error: Fallo al guardar clase " << modelo.clases[i] << std::endl;
            return false;
        }
    }

    std::cout << "   & Modelo modular guardado: " << modelo.clases.size() << " clases";
    std::cout << " en " << ruta_base << std::endl;

    return true;
}

/**
 * Carga un modelo SVM completo desde formato modular
 * Lee metadata.json y todos los archivos class_*.bin
 *
 * @param ruta_base Directorio base (ej: "model/")
 * @return ModeloSVM cargado (vac�o si hay error)
 */
ModeloSVM cargarModeloModular(const std::string& ruta_base) {
    ModeloSVM modelo;

    std::cout << "-> Cargando modelo SVM modular: " << ruta_base << std::endl;

    // Cargar metadata
    int num_clases, dimension;
    std::vector<int> clases;

    if (!cargarMetadata(ruta_base, num_clases, dimension, clases)) {
        std::cerr << "! Error: No se pudo cargar metadata" << std::endl;
        return modelo;
    }

    modelo.dimensionCaracteristicas = dimension;
    modelo.clases = clases;
    modelo.pesosPorClase.resize(num_clases);
    modelo.biasPorClase.resize(num_clases);
    modelo.plattAPorClase.resize(num_clases);
    modelo.plattBPorClase.resize(num_clases);
    modelo.thresholdsPorClase.resize(num_clases);

    // Cargar cada clasificador binario
    for (size_t i = 0; i < clases.size(); ++i) {
        int clase = clases[i];
        ClasificadorBinario clasificador;

        if (!cargarClasificadorBinario(ruta_base, clase, clasificador)) {
            std::cerr << "! Error: Fallo al cargar clase " << clase << std::endl;
            return ModeloSVM();  // Retornar modelo vacio
        }

        // Guardar en estructuras del modelo
        modelo.pesosPorClase[i] = clasificador.pesos;
        modelo.biasPorClase[i] = clasificador.bias;
        modelo.plattAPorClase[i] = clasificador.plattA;
        modelo.plattBPorClase[i] = clasificador.plattB;
        modelo.thresholdsPorClase[i] = clasificador.thresholdOptimo;
    }

    std::cout << "   & Modelo modular cargado: " << num_clases << " clases, "
        << dimension << " caracteristicas" << std::endl;

    return modelo;
}
