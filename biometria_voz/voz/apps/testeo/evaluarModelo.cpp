#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <random>
#include "../../core/classification/svm.h"
#include "../../core/classification/metrics/svm_metrics.h"
#include "../../core/pipeline/audio_pipeline.h"
#include "../../utils/config.h"

namespace fs = std::filesystem;

// CONFIGURACION DE EVALUACION

struct ConfigEvaluacion {
    int maxAudiosPorClase = 10;        // Numero maximo de audios a procesar por clase (0 = todos)
    bool seleccionAleatoria = true;    // Si es true, selecciona aleatoriamente; si es false, toma los primeros
    unsigned int seed = 42;             // Semilla para reproducibilidad
};

// ESTRUCTURA PARA RESULTADOS DE EVALUACION

struct ResultadosEvaluacion {
    int totalAudios = 0;
    int totalCorrectos = 0;
    int totalIncorrectos = 0;
    AudioSample accuracyGlobal = 0.0;
    
    std::vector<int> y_real;            // Etiquetas reales
    std::vector<int> y_pred;            // Etiquetas predichas
    std::vector<int> clases;            // Lista de clases
    
    std::map<int, std::string> nombreClase;  // ID -> Nombre carpeta
    
    AudioSample tiempoPromedioMs = 0.0;
};

// FUNCIONES AUXILIARES

/**
 * Obtiene lista de archivos de audio en un directorio
 * Si maxAudios > 0, selecciona aleatoriamente ese numero
 */
std::vector<std::string> obtenerArchivosAudio(const std::string& directorio, 
                                               const ConfigEvaluacion& config) {
    std::vector<std::string> archivos;
    
    if (!fs::exists(directorio) || !fs::is_directory(directorio)) {
        return archivos;
    }
    
    const std::vector<std::string> extensionesValidas = {".wav", ".mp3", ".flac", ".aiff"};
    
    // Recolectar todos los archivos de audio
    for (const auto& entry : fs::directory_iterator(directorio)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(extensionesValidas.begin(), extensionesValidas.end(), ext) != 
                extensionesValidas.end()) {
                archivos.push_back(entry.path().string());
            }
        }
    }
    
    // Seleccion aleatoria si esta habilitada y hay limite
    if (config.seleccionAleatoria && config.maxAudiosPorClase > 0 && 
        archivos.size() > static_cast<size_t>(config.maxAudiosPorClase)) {
        
        std::mt19937 rng(config.seed);
        std::shuffle(archivos.begin(), archivos.end(), rng);
        archivos.resize(config.maxAudiosPorClase);
    }
    // Si no es aleatorio pero hay limite, tomar los primeros
    else if (!config.seleccionAleatoria && config.maxAudiosPorClase > 0 && 
             archivos.size() > static_cast<size_t>(config.maxAudiosPorClase)) {
        archivos.resize(config.maxAudiosPorClase);
    }
    
    return archivos;
}

/**
 * Obtiene el ID de clase a partir del nombre de carpeta
 * Ejemplo: "speaker_001" -> 1
 */
int extraerIdClase(const std::string& nombreCarpeta) {
    // Buscar el ultimo numero en el nombre de carpeta
    std::string numStr;
    bool encontrado = false;
    
    for (char c : nombreCarpeta) {
        if (std::isdigit(c)) {
            numStr += c;
            encontrado = true;
        } else if (encontrado && !numStr.empty()) {
            break;  // Terminar si ya encontramos numeros y ahora hay letras
        }
    }
    
    if (!numStr.empty()) {
        return std::stoi(numStr);
    }
    
    return -1;  // No se pudo extraer ID
}

/**
 * Procesa un audio y obtiene el vector de caracteristicas
 */
bool procesarUnAudio(const std::string& audioPath, std::vector<AudioSample>& features) {
    std::vector<std::vector<AudioSample>> featuresMultiples;
    
    // Desactivar augmentation temporalmente para evaluacion
    bool augmentacionOriginal = CONFIG_DATASET.usarAugmentation;
    CONFIG_DATASET.usarAugmentation = false;
    
    bool resultado = procesarAudioCompleto(audioPath, featuresMultiples);
    
    // Restaurar configuracion
    CONFIG_DATASET.usarAugmentation = augmentacionOriginal;
    
    if (resultado && !featuresMultiples.empty()) {
        features = featuresMultiples[0];  // Solo tomamos el primer vector
        
        // NOTA: NO aplicar expansion polinomial aqui
        // El pipeline (audio_pipeline.cpp) ya la aplica automaticamente
        // si CONFIG_SVM.usarExpansionPolinomial esta activado
        
        return true;
    }
    
    return false;
}

/**
 * Imprime metricas detalladas usando funciones del modulo de clasificacion
 */
void imprimirMetricasDetalladas(const ResultadosEvaluacion& resultados) {
    std::cout << "\n=== MATRIZ DE CONFUSION MULTICLASE ===" << std::endl;
    auto matrizMulti = calcularMatrizConfusionMulticlase(resultados.y_real, resultados.y_pred);
    mostrarMatrizConfusionMulticlase(matrizMulti, resultados.clases);
    
    std::cout << "\n=== METRICAS POR CLASE ===" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::left 
              << std::setw(8) << "Clase"
              << std::setw(20) << "Nombre"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall"
              << std::setw(12) << "F1-Score"
              << std::setw(12) << "Specificity"
              << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (int clase : resultados.clases) {
        Metricas m = calcularMetricas(resultados.y_real, resultados.y_pred, clase);
        std::string nombre = resultados.nombreClase.count(clase) ? 
                            resultados.nombreClase.at(clase) : "Desconocido";
        
        std::cout << std::left 
                  << std::setw(8) << clase
                  << std::setw(20) << nombre
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << (m.precision * 100.0)
                  << std::setw(12) << (m.recall * 100.0)
                  << std::setw(12) << (m.f1_score * 100.0)
                  << std::setw(12) << (m.specificity * 100.0)
                  << std::endl;
    }
    
    // Estadisticas multiclase (macro-average)
    EstadisticasMulticlase stats = calcularEstadisticasMulticlase(
        resultados.y_real, resultados.y_pred, resultados.clases
    );
    
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::left 
              << std::setw(28) << "PROMEDIO (Macro)"
              << std::fixed << std::setprecision(2)
              << std::setw(12) << (stats.precision_promedio * 100.0)
              << std::setw(12) << (stats.recall_promedio * 100.0)
              << std::setw(12) << (stats.f1_promedio * 100.0)
              << std::setw(12) << (stats.specificity_promedio * 100.0)
              << std::endl;
    std::cout << std::string(90, '-') << std::endl;
}

// FUNCION PRINCIPAL DE EVALUACION

/**
 * Evalua el modelo con audios de una carpeta estructurada por clases
 * 
 * Estructura esperada:
 *   directorio_base/
 *     ├── clase_1/
 *     │   ├── audio1.wav
 *     │   ├── audio2.wav
 *     │   └── ...
 *     ├── clase_2/
 *     │   ├── audio1.wav
 *     │   └── ...
 *     └── ...
 */
ResultadosEvaluacion evaluarModelo(const ModeloSVM& modelo, 
                                    const std::string& directorioBase,
                                    const ConfigEvaluacion& config) {
    ResultadosEvaluacion resultados;
    
    std::cout << "\n=== EVALUANDO MODELO ===" << std::endl;
    std::cout << "Directorio: " << directorioBase << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Verificar que el directorio existe
    if (!fs::exists(directorioBase) || !fs::is_directory(directorioBase)) {
        std::cout << "\n% ERROR: Directorio no existe o no es valido" << std::endl;
        return resultados;
    }
    
    // Recolectar carpetas de clases
    std::vector<std::pair<int, std::string>> clasesEncontradas;  // (ID, ruta)
    
    std::cout << "\n@ Escaneando carpetas de clases..." << std::endl;
    for (const auto& entry : fs::directory_iterator(directorioBase)) {
        if (entry.is_directory()) {
            std::string nombreCarpeta = entry.path().filename().string();
            int idClase = extraerIdClase(nombreCarpeta);
            
            if (idClase >= 0) {
                clasesEncontradas.push_back({idClase, entry.path().string()});
                resultados.nombreClase[idClase] = nombreCarpeta;
                std::cout << "  -> Clase " << idClase << ": " << nombreCarpeta << std::endl;
            } else {
                std::cout << "  * Ignorando carpeta: " << nombreCarpeta 
                          << " (no se pudo extraer ID)" << std::endl;
            }
        }
    }
    
    if (clasesEncontradas.empty()) {
        std::cout << "\n% ERROR: No se encontraron carpetas de clases validas" << std::endl;
        std::cout << "  Asegurate de que las carpetas tengan formato: clase_XXX o similar" << std::endl;
        return resultados;
    }
    
    std::cout << "\n  Total clases encontradas: " << clasesEncontradas.size() << std::endl;
    
    // Almacenar clases para usar en metricas
    for (const auto& [id, _] : clasesEncontradas) {
        resultados.clases.push_back(id);
    }
    std::sort(resultados.clases.begin(), resultados.clases.end());
    
    // Procesar cada clase
    std::vector<double> tiemposProcesamiento;
    
    std::cout << "\n@ Procesando audios..." << std::endl;
    if (config.maxAudiosPorClase > 0) {
        std::cout << "  Limite por clase: " << config.maxAudiosPorClase << " audios" << std::endl;
        std::cout << "  Seleccion: " << (config.seleccionAleatoria ? "Aleatoria" : "Primeros N") << std::endl;
        if (config.seleccionAleatoria) {
            std::cout << "  Semilla: " << config.seed << std::endl;
        }
    } else {
        std::cout << "  Procesando TODOS los audios de cada clase" << std::endl;
    }
    std::cout << std::endl;
    
    for (const auto& [idClaseReal, rutaCarpeta] : clasesEncontradas) {
        std::vector<std::string> archivos = obtenerArchivosAudio(rutaCarpeta, config);
        
        if (archivos.empty()) {
            std::cout << "  * Clase " << idClaseReal << ": Sin archivos de audio" << std::endl;
            continue;
        }
        
        std::cout << "  -> Clase " << idClaseReal << ": " << archivos.size() 
                  << " audios encontrados" << std::endl;
        
        int procesados = 0;
        int correctos = 0;
        
        for (const auto& archivoPath : archivos) {
            auto inicio = std::chrono::high_resolution_clock::now();
            
            // Procesar audio
            std::vector<AudioSample> features;
            if (!procesarUnAudio(archivoPath, features)) {
                std::cout << "     * Error procesando: " 
                          << fs::path(archivoPath).filename().string() << std::endl;
                continue;
            }
            
            // NOTA: La normalizacion L2 ya se aplica en el pipeline automaticamente
            //       si CONFIG_SVM.usarNormalizacionL2 == true
            
            // Predecir
            int idPredicho = predecirHablante(features, modelo);
            
            auto fin = std::chrono::high_resolution_clock::now();
            auto duracion = std::chrono::duration_cast<std::chrono::microseconds>(fin - inicio);
            tiemposProcesamiento.push_back(duracion.count() / 1000.0);  // Convertir a ms
            
            // Actualizar estadisticas
            procesados++;
            resultados.totalAudios++;
            
            bool correcto = (idPredicho == idClaseReal);
            if (correcto) {
                correctos++;
                resultados.totalCorrectos++;
            } else {
                resultados.totalIncorrectos++;
            }
            
            // Almacenar para calcular metricas despues
            resultados.y_real.push_back(idClaseReal);
            resultados.y_pred.push_back(idPredicho);
        }
        
        AudioSample accuracyClase = procesados > 0 ? 
            (static_cast<AudioSample>(correctos) / procesados * 100.0) : 0.0;
        
        std::cout << "     Procesados: " << procesados 
                  << " | Correctos: " << correctos 
                  << " | Accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracyClase << "%" << std::endl;
    }
    
    // Calcular metricas globales
    if (resultados.totalAudios > 0) {
        resultados.accuracyGlobal = 
            static_cast<AudioSample>(resultados.totalCorrectos) / resultados.totalAudios;
    }
    
    if (!tiemposProcesamiento.empty()) {
        resultados.tiempoPromedioMs = 
            std::accumulate(tiemposProcesamiento.begin(), tiemposProcesamiento.end(), 0.0) / 
            tiemposProcesamiento.size();
    }
    
    return resultados;
}

// MAIN

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "     EVALUACION DE MODELO SVM - SISTEMA BIOMETRICO DE VOZ" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Ruta del modelo
    std::string modelPath = obtenerRutaModelo();
    
    std::cout << "\n@ Configuracion" << std::endl;
    std::cout << "  Ruta modelo: " << modelPath << std::endl;
    
    // Verificar que el modelo existe
    bool esModular = fs::is_directory(modelPath);
    if (!esModular && !fs::exists(modelPath)) {
        std::cout << "\n% ERROR: Modelo no encontrado" << std::endl;
        std::cout << "  Ruta: " << modelPath << std::endl;
        std::cout << "  Entrena un modelo primero usando 'entrenar_modelo'" << std::endl;
        return 1;
    }
    
    // Cargar modelo
    std::cout << "\n@ Cargando modelo..." << std::endl;
    ModeloSVM modelo;
    
    if (esModular) {
        std::cout << "  Formato: MODULAR (directorio)" << std::endl;
        modelo = cargarModeloModular(modelPath);
    } else {
        std::cout << "  Formato: MONOLITICO (archivo unico)" << std::endl;
        modelo = cargarModeloSVM(modelPath);
    }
    
    if (modelo.clases.empty()) {
        std::cout << "\n% ERROR: Modelo no valido o corrupto" << std::endl;
        return 1;
    }
    
    std::cout << "  -> Clases en modelo: " << modelo.clases.size() << std::endl;
    std::cout << "  -> Dimension features: " << modelo.dimensionCaracteristicas << std::endl;
    std::cout << "  -> Normalizacion L2: " << (CONFIG_SVM.usarNormalizacionL2 ? "ACTIVADA" : "DESACTIVADA") << std::endl;
    
    // Ruta de audios de prueba
   // std::string directorioAudios = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V3\\train";
   // std::string directorioAudios = "D:\\Dataset\\audio7";
    // std::string directorioAudios = "D:\\Dataset_test";
     // std::string directorioAudios = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V1\\mls_spanish\\train\\audio";
      std::string directorioAudios = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V1\\mls_spanish\\audio";
   // std::string directorioAudios = "D:\\8vo-Nivel\\Tesiss\\DatasetReal";

    std::cout << "\n@ Directorio de audios" << std::endl;
    std::cout << "  Ruta: " << directorioAudios << std::endl;
    
    // Configuracion de evaluacion
    ConfigEvaluacion config;
    config.maxAudiosPorClase = 40;     // Procesar maximo 20 audios por clase
    config.seleccionAleatoria = true;   // Seleccion aleatoria
    config.seed = 42;                   // Semilla para reproducibilidad
    
    std::cout << "\n@ Configuracion de evaluacion" << std::endl;
    std::cout << "  Max audios por clase: " << config.maxAudiosPorClase << std::endl;
    std::cout << "  Seleccion aleatoria: " << (config.seleccionAleatoria ? "SI" : "NO") << std::endl;
    std::cout << "  Semilla: " << config.seed << std::endl;
    
    // Evaluar
    ResultadosEvaluacion resultados = evaluarModelo(modelo, directorioAudios, config);
    
    if (resultados.totalAudios == 0) {
        std::cout << "\n% No se procesaron audios" << std::endl;
        return 1;
    }
    
    // Imprimir resultados
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "=== RESULTADOS FINALES ===" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "\n# Estadisticas Globales" << std::endl;
    std::cout << "  Total audios procesados: " << resultados.totalAudios << std::endl;
    std::cout << "  Predicciones correctas:  " << resultados.totalCorrectos << std::endl;
    std::cout << "  Predicciones incorrectas: " << resultados.totalIncorrectos << std::endl;
    std::cout << "  Accuracy global:         " << std::fixed << std::setprecision(4) 
              << (resultados.accuracyGlobal * 100.0) << "%" << std::endl;
    std::cout << "  Tiempo promedio/audio:   " << std::fixed << std::setprecision(2) 
              << resultados.tiempoPromedioMs << " ms" << std::endl;
    
    // Metricas detalladas usando funciones del modulo de clasificacion
    imprimirMetricasDetalladas(resultados);
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Evaluacion completada exitosamente" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    return 0;
}
