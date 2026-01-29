#include "utilidades/svm_ova_utils.h"
#include "svm/svm_entrenamiento.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>

void dilatacion3x3_binaria(uint8_t* data, int ancho, int alto) {
    auto salida = std::make_unique<uint8_t[]>(ancho * alto);
    std::fill(salida.get(), salida.get() + ancho * alto, 0);

    for (int y = 1; y < alto - 1; ++y) {
        for (int x = 1; x < ancho - 1; ++x) {
            bool hayBlanco = false;
            for (int j = -1; j <= 1 && !hayBlanco; ++j)
                for (int i = -1; i <= 1; ++i)
                    if (data[(y + j) * ancho + (x + i)] == 255)
                        hayBlanco = true;
            salida[y * ancho + x] = hayBlanco ? 255 : 0;
        }
    }

    std::copy(salida.get(), salida.get() + (ancho * alto), data);
}

// Carga el modelo SVM OVA desde archivo
bool cargarModeloSVM(const std::string& ruta, ModeloSVM& modelo) {
    std::ifstream in(ruta, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "No se pudo abrir el archivo binario del modelo SVM: " << ruta << "\n";
        return false;
    }

    modelo.clases.clear();
    modelo.pesosPorClase.clear();
    modelo.biasPorClase.clear();

    size_t numClases;
    in.read(reinterpret_cast<char*>(&numClases), sizeof(size_t));
    if (in.fail()) {
        std::cerr << "Error leyendo numero de clases\n";
        return false;
    }

    for (size_t i = 0; i < numClases; ++i) {
        int claseId;
        in.read(reinterpret_cast<char*>(&claseId), sizeof(int));
        if (in.fail()) {
            std::cerr << "Error leyendo ID de clase " << i << "\n";
            return false;
        }

        size_t tam;
        in.read(reinterpret_cast<char*>(&tam), sizeof(size_t));
        if (in.fail() || tam == 0 || tam > 10000) {
            std::cerr << "Tamaño inválido de vector de pesos para clase " << i << ": " << tam << "\n";
            return false;
        }

        std::vector<double> pesos(tam);
        in.read(reinterpret_cast<char*>(pesos.data()), tam * sizeof(double));
        if (in.fail()) {
            std::cerr << "Error leyendo pesos de clase " << i << "\n";
            return false;
        }

        double bias;
        in.read(reinterpret_cast<char*>(&bias), sizeof(double));
        if (in.fail()) {
            std::cerr << "Error leyendo bias de clase " << i << "\n";
            return false;
        }

        modelo.clases.push_back(claseId);
        modelo.pesosPorClase.push_back(std::move(pesos));
        modelo.biasPorClase.push_back(bias);
    }

    return true;
}

bool guardarModeloSVM(const std::string& ruta, const ModeloSVM& modelo) {
    std::ofstream out(ruta, std::ios::binary);
    if (!out.is_open()) return false;

    size_t numClases = modelo.clases.size();
    out.write((char*)&numClases, sizeof(size_t));
    for (size_t i = 0; i < numClases; ++i) {
        int c = modelo.clases[i];
        out.write((char*)&c, sizeof(int));

        size_t tam = modelo.pesosPorClase[i].size();
        out.write((char*)&tam, sizeof(size_t));
        out.write((char*)modelo.pesosPorClase[i].data(), tam * sizeof(double));
        out.write((char*)&modelo.biasPorClase[i], sizeof(double));
    }
    return true;
}