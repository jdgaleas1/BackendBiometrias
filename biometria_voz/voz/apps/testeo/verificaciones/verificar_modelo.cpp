#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <filesystem>
#include "../../../core/classification/svm.h"
#include "../../../utils/config.h"

namespace fs = std::filesystem;

void verificarModelo() {
    // Detectar tipo de modelo: monolitico (.svm) o modular (directorio)
    std::string modelPath = obtenerRutaModelo();
    bool esModular = false;
    ModeloSVM modelo;

    std::cout << "=== VERIFICACION DEL MODELO SVM ===" << std::endl;
    std::cout << "Ruta configurada: " << modelPath << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Verificar si es un directorio (modelo modular)
    if (fs::is_directory(modelPath)) {
        std::cout << "\n@ Detectado formato MODULAR (directorio)" << std::endl;
        esModular = true;

        // Verificar metadata.json
        std::string metadataPath = modelPath + "/metadata.json";
        if (!fs::exists(metadataPath)) {
            std::cout << "% ERROR: No existe " << metadataPath << std::endl;
            std::cout << "   El directorio no contiene un modelo valido" << std::endl;
            std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
            std::cin.get();
            return;
        }

        // Contar archivos .bin
        int numClasificadores = 0;
        for (const auto& entry : fs::directory_iterator(modelPath)) {
            if (entry.path().extension() == ".bin") {
                numClasificadores++;
            }
        }

        std::cout << "  Metadata: " << metadataPath << std::endl;
        std::cout << "  Clasificadores encontrados: " << numClasificadores << " archivos .bin" << std::endl;

        // Cargar modelo modular
        std::cout << "\n@ Cargando modelo modular..." << std::endl;
        modelo = cargarModeloModular(modelPath);
        
        if (!modelo.clases.empty()) {
            // Verificar consistencia entre archivos .bin y metadata
            if (numClasificadores != static_cast<int>(modelo.clases.size())) {
                std::cout << "\n% ADVERTENCIA: Inconsistencia detectada" << std::endl;
                std::cout << "   Archivos .bin: " << numClasificadores << std::endl;
                std::cout << "   Clases en metadata.json: " << modelo.clases.size() << std::endl;
                if (numClasificadores > static_cast<int>(modelo.clases.size())) {
                    std::cout << "   -> Hay archivos .bin huerfanos (no listados en metadata)" << std::endl;
                } else {
                    std::cout << "   -> Faltan archivos .bin para algunas clases" << std::endl;
                }
            }
        }
    }
    else {
        // Modelo monolitico (.svm)
        std::cout << "\n@ Detectado formato MONOLITICO (archivo unico)" << std::endl;

        // Verificar que el archivo existe
        std::ifstream check(modelPath, std::ios::binary | std::ios::ate);
        if (!check.is_open()) {
            std::cout << "% ERROR: No se puede abrir: " << modelPath << std::endl;
            std::cout << "   Ruta absoluta: " << fs::absolute(modelPath) << std::endl;
            std::cout << "   Verifica que el archivo existe" << std::endl;
            std::cout << "   ¿Has entrenado el modelo?" << std::endl;
            std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
            std::cin.get();
            return;
        }

        std::streamsize fileSize = check.tellg();
        std::cout << "  Archivo: " << modelPath << std::endl;
        std::cout << "  Tamaño: " << (fileSize / 1024.0) << " KB (" << fileSize << " bytes)" << std::endl;

        if (fileSize < 100) {
            std::cout << "  % ADVERTENCIA: Archivo muy pequeño (< 100 bytes)" << std::endl;
            std::cout << "  El modelo podria estar vacio o corrupto" << std::endl;
        }
        check.close();

        // Cargar modelo monolitico
        std::cout << "\n@ Cargando modelo monolitico..." << std::endl;
        modelo = cargarModeloSVM(modelPath);
    }
    
    if (modelo.clases.empty()) {
        std::cout << "\n% ERROR: No se pudo cargar el modelo" << std::endl;
        std::cout << "   El archivo/directorio podria estar corrupto" << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }

    // Validaciones del modelo
    bool modeloValido = true;
    
    if (modelo.clases.size() != modelo.pesosPorClase.size()) {
        std::cout << "% ERROR: Desincronizacion entre clases y pesos" << std::endl;
        std::cout << "   Clases: " << modelo.clases.size() << ", Pesos: " << modelo.pesosPorClase.size() << std::endl;
        modeloValido = false;
    }
    
    if (modelo.clases.size() != modelo.biasPorClase.size()) {
        std::cout << "% ERROR: Desincronizacion entre clases y bias" << std::endl;
        std::cout << "   Clases: " << modelo.clases.size() << ", Bias: " << modelo.biasPorClase.size() << std::endl;
        modeloValido = false;
    }
    
    if (modelo.dimensionCaracteristicas == 0) {
        std::cout << "% ERROR: Dimension de caracteristicas es 0" << std::endl;
        modeloValido = false;
    }
    
    // Verificar que los pesos tienen la dimension correcta
    for (size_t i = 0; i < modelo.pesosPorClase.size(); ++i) {
        if (modelo.pesosPorClase[i].size() != modelo.dimensionCaracteristicas) {
            std::cout << "% ERROR: Clase " << modelo.clases[i] << " tiene " 
                      << modelo.pesosPorClase[i].size() << " pesos (esperados: " 
                      << modelo.dimensionCaracteristicas << ")" << std::endl;
            modeloValido = false;
            break;
        }
    }

    if (!modeloValido) {
        std::cout << "\n% El modelo tiene inconsistencias internas" << std::endl;
        std::cout << "   Reentrenar el modelo es recomendado" << std::endl;
        std::cout << "   Presiona cualquier tecla para continuar..." << std::endl;
        std::cin.get();
        return;
    }

    std::cout << "\n@ Modelo cargado exitosamente" << std::endl;
    std::cout << "  Tipo: " << (esModular ? "MODULAR" : "MONOLITICO") << std::endl;
    std::cout << "  Numero de clases: " << modelo.clases.size() << std::endl;
    std::cout << "  Dimension caracteristicas: " << modelo.dimensionCaracteristicas << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Crear vector con clase y bias para ordenar
    std::vector<std::pair<AudioSample, int>> biasOrdenado;  // AudioSample para bias
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        biasOrdenado.push_back({ modelo.biasPorClase[i], modelo.clases[i] });
    }

    // Ordenar por bias descendente
    std::sort(biasOrdenado.begin(), biasOrdenado.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "\n=== ANALISIS DE BIAS POR CLASE (Mayor a Menor) ===" << std::endl;
    std::cout << "Rank | Clase | Bias" << std::endl;
    std::cout << std::string(30, '-') << std::endl;

    AudioSample minBias = biasOrdenado.back().first;   // El ultimo (menor)
    AudioSample maxBias = biasOrdenado.front().first;  // El primero (mayor)
    int countBiasAltos = 0;      // Bias > 0
    int countBiasBajos = 0;      // Bias < -3.0 (muy negativo)
    int countBiasCero = 0;       // Bias cerca de 0 (abs < 0.5)
    int countBiasIguales = 0;    // Clases que no se entrenaron (bias = -3.891820)
    
    for (size_t i = 0; i < biasOrdenado.size(); ++i) {
        AudioSample bias = biasOrdenado[i].first;
        int clase = biasOrdenado[i].second;
        
        std::cout << std::setw(4) << (i + 1) << " | "
            << std::setw(5) << clase << " | "
            << std::setw(8) << std::fixed << std::setprecision(4) << bias;
        
        if (bias > 0.0) {
            std::cout << " <- @ Clasificador confiado";
            countBiasAltos++;
        } else if (bias < -3.0) {
            std::cout << " <- % Clasificador muy conservador";
            countBiasBajos++;
        } else if (std::abs(bias) < 0.5) {
            std::cout << " <- @ Cerca del umbral";
            countBiasCero++;
        }
        
        // Detectar clases que no se entrenaron (valor especifico del inicializador)
        if (std::abs(bias - (-3.891820)) < 0.000001) {
            std::cout << " <- % NO ENTRENADO";
            countBiasIguales++;
        }
        
        std::cout << std::endl;
    }

    std::cout << "\n=== ANALISIS DE PESOS POR CLASE ===" << std::endl;
    std::cout << "Clase | Norma^2 | ||w|| | Estado" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // CAMBIO CRITICO: Usar AudioSample (double) para calculos de norma
    AudioSample minNorm = 1e9, maxNorm = 0.0;
    int countNormBaja = 0;  // Norma < 0.05 (sospechoso)
    int countNormAlta = 0;  // Norma > 0.5
    int countPesosProblematicos = 0;
    
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        // PRECISION COMPLETA: AudioSample (double) para calculo de norma
        AudioSample normSq = 0.0;
        int countNaN = 0, countInf = 0, countZero = 0;
        
        // CORRECCION: Iterar con AudioSample, no float
        for (AudioSample w : modelo.pesosPorClase[i]) {
            if (std::isnan(w)) countNaN++;
            else if (std::isinf(w)) countInf++;
            else if (w == 0.0) countZero++;
            normSq += w * w;  // Acumulacion en DOUBLE precision (~1e-15)
        }
        
        AudioSample norm = std::sqrt(normSq);  // Resultado en DOUBLE
        
        if (norm < minNorm) minNorm = norm;
        if (norm > maxNorm) maxNorm = norm;
        if (norm < 0.05) countNormBaja++;
        if (norm > 0.5) countNormAlta++;
        if (countNaN > 0 || countInf > 0) countPesosProblematicos++;
        
        std::cout << std::setw(5) << modelo.clases[i] << " | "
            << std::setw(8) << std::fixed << std::setprecision(4) << normSq << " | "
            << std::setw(8) << norm << " | ";
        
        if (countNaN > 0) std::cout << "NaN:" << countNaN << " ";
        if (countInf > 0) std::cout << "Inf:" << countInf << " ";
        if (countZero == static_cast<int>(modelo.pesosPorClase[i].size())) 
            std::cout << "% TODOS CERO ";
        
        if (norm < 0.05) {
            std::cout << " <- % Pesos muy pequenos";
        } else if (norm > 0.5) {
            std::cout << " <- @ Pesos grandes";
        }
        std::cout << std::endl;
    }
    
    if (countPesosProblematicos > 0) {
        std::cout << "\n% ADVERTENCIA: " << countPesosProblematicos 
                  << " clases tienen pesos NaN o Infinitos" << std::endl;
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "=== DIAGNOSTICO DEL MODELO ===" << std::endl;
    std::cout << "\nBIAS:" << std::endl;
    std::cout << "  Rango: [" << std::fixed << std::setprecision(2)
        << minBias << ", " << maxBias << "]" << std::endl;
    std::cout << "  Diferencia max-min: " << (maxBias - minBias) << std::endl;
    std::cout << "  Clases con bias > 0: " << countBiasAltos << " de " << modelo.clases.size() 
              << " (" << (100.0 * countBiasAltos / modelo.clases.size()) << "%)" << std::endl;
    std::cout << "  Clases con bias < -3.0: " << countBiasBajos << " de " << modelo.clases.size() 
              << " (" << (100.0 * countBiasBajos / modelo.clases.size()) << "%)" << std::endl;
    std::cout << "  Clases cercanas a 0: " << countBiasCero << " de " << modelo.clases.size() << std::endl;
    std::cout << "  Clases NO entrenadas (bias = -3.891820): " << countBiasIguales << " de " << modelo.clases.size();
    if (countBiasIguales > 0) {
        std::cout << " % PROBLEMA DETECTADO";
    }
    std::cout << std::endl;
    
    std::cout << "\nPESOS (Normas):" << std::endl;
    std::cout << "  Rango ||w||: [" << std::fixed << std::setprecision(4)
        << minNorm << ", " << maxNorm << "]" << std::endl;
    std::cout << "  Clases con ||w|| < 0.05: " << countNormBaja << " de " << modelo.clases.size() 
              << " (" << (100.0 * countNormBaja / modelo.clases.size()) << "%)" << std::endl;
    std::cout << "  Clases con ||w|| > 0.5: " << countNormAlta << " de " << modelo.clases.size() 
              << " (" << (100.0 * countNormAlta / modelo.clases.size()) << "%)" << std::endl;
    if (countPesosProblematicos > 0) {
        std::cout << "  % Clases con pesos NaN/Inf: " << countPesosProblematicos << std::endl;
    }
    std::cout << "  @ Precision calculos: AudioSample (double, ~1e-15)" << std::endl;

    // Recomendaciones basadas en el analisis
    std::cout << "\n-> RECOMENDACIONES:" << std::endl;
    
    // Detectar clases con pesos en cero (CRITICO)
    std::vector<int> clasesConPesosCero;
    for (size_t i = 0; i < modelo.clases.size(); ++i) {
        // CORREGIDO: Calculo en AudioSample (double precision)
        AudioSample norma = 0.0;
        for (AudioSample w : modelo.pesosPorClase[i]) {
            norma += w * w;
        }
        if (norma == 0.0) {
            clasesConPesosCero.push_back(modelo.clases[i]);
        }
    }
    
    if (!clasesConPesosCero.empty()) {
        std::cout << "   % CRITICO: " << clasesConPesosCero.size() << " clases con pesos en CERO:" << std::endl;
        std::cout << "      Clases afectadas: ";
        for (size_t i = 0; i < clasesConPesosCero.size(); ++i) {
            std::cout << clasesConPesosCero[i];
            if (i < clasesConPesosCero.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "   -> PROBLEMA: Estas clases NO aprendieron nada durante el entrenamiento" << std::endl;
        std::cout << "   -> CAUSAS POSIBLES:" << std::endl;
        std::cout << "      1. Dataset muy pequeno para estas clases (< 10 muestras)" << std::endl;
        std::cout << "      2. Todas las muestras identicas (varianza cero)" << std::endl;
        std::cout << "      3. Early stopping se activo antes de aprender" << std::endl;
        if (esModular) {
            std::cout << "   -> SOLUCION: Reentrenar solo esas clases:" << std::endl;
            std::cout << "      - Verificar que tengan suficientes muestras variadas" << std::endl;
            std::cout << "      - Ajustar CONFIG_SVM.maxIteraciones o specificityObjetivo" << std::endl;
        } else {
            std::cout << "   -> SOLUCION: Reentrenar el modelo completo verificando el dataset" << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (countBiasIguales > static_cast<int>(modelo.clases.size()) / 2) {
        std::cout << "   % CRITICO: Mas de " << countBiasIguales << " clases NO SE ENTRENARON" << std::endl;
        std::cout << "   -> El codigo viejo sigue compilado o hubo error en entrenamiento" << std::endl;
        std::cout << "   -> SOLUCION: Borrar build/ y recompilar desde cero" << std::endl;
    }
    else if (countBiasIguales > 0) {
        std::cout << "   % ADVERTENCIA: " << countBiasIguales << " clases no se entrenaron completamente" << std::endl;
        std::cout << "   -> Revisar logs de entrenamiento para esas clases" << std::endl;
        if (esModular) {
            std::cout << "   -> Puedes reentrenar solo esas clases con entrenamiento incremental" << std::endl;
        }
    }
    else if (countNormBaja > static_cast<int>(modelo.clases.size()) / 2 && clasesConPesosCero.empty()) {
        std::cout << "   % ADVERTENCIA: Mas de " << countNormBaja << " clases con pesos muy pequenos" << std::endl;
        std::cout << "   -> Posible problema: Learning rate demasiado bajo o regularizacion muy alta" << std::endl;
        std::cout << "   -> Revisar config.h: tasaAprendizaje y C" << std::endl;
    }
    else if (countBiasAltos == 0 && modelo.clases.size() > 10) {
        std::cout << "   @ INFO: TODAS las clases tienen bias negativo" << std::endl;
        std::cout << "   -> Esto indica que el clasificador es muy conservador" << std::endl;
        std::cout << "   -> Dataset posiblemente muy desbalanceado o dificil de separar" << std::endl;
        std::cout << "   -> Clases con bias menos negativo son las mas faciles de identificar" << std::endl;
    }
    else if (maxBias - minBias > 10.0) {
        std::cout << "   @ INFO: Gran variacion en bias (" << (maxBias - minBias) << ")" << std::endl;
        std::cout << "   -> Esto es NORMAL en datasets desbalanceados" << std::endl;
        std::cout << "   -> Clases con bias alto: mas faciles de identificar" << std::endl;
        std::cout << "   -> Clases con bias bajo: mas dificiles de separar" << std::endl;
    }
    else if (clasesConPesosCero.empty()) {
        std::cout << "   @ El modelo parece entrenado correctamente" << std::endl;
        std::cout << "   -> Bias distribuido: " << countBiasAltos << " positivos, "
                  << (modelo.clases.size() - countBiasAltos) << " negativos" << std::endl;
        std::cout << "   -> Pesos activos: " << (modelo.clases.size() - countNormBaja) 
                  << " clases con ||w|| >= 0.05" << std::endl;
    }

    // Informacion adicional segun el formato
    if (esModular) {
        std::cout << "\n-> INFO DEL FORMATO MODULAR:" << std::endl;
        std::cout << "   @ Cada clase tiene su propio archivo .bin" << std::endl;
        std::cout << "   @ Puedes agregar nuevas clases sin reentrenar todo" << std::endl;
        std::cout << "   @ Archivos en: " << modelPath << "/" << std::endl;
    }

    std::cout << std::string(70, '=') << std::endl;
}

int main() {
    verificarModelo();

    std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
    std::cin.get();

    return 0;
}