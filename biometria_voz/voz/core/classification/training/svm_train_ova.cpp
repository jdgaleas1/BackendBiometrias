// svm_train_ova.cpp - WRAPPER PARA ENTRENAMIENTO ONE-VS-ALL
// Este archivo solo coordina el entrenamiento de multiples clasificadores binarios

#include "svm_training.h"
#include "../../../utils/config.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <cstdlib>

// Declaraciones de funciones auxiliares
std::vector<int> prepararDatosBinarios(
    const std::vector<int>& y,
    int clase_positiva,
    int& positivas,
    int& negativas
);

// ============================================================================
// ENTRENAMIENTO ONE-VS-ALL MULTICLASE
// ============================================================================

ModeloSVM entrenarSVMOVA(const std::vector<std::vector<AudioSample>>& X,
                         const std::vector<int>& y) {
    
    auto& cfg = CONFIG_SVM;
    
    int m = static_cast<int>(X.size());
    int n = static_cast<int>(X[0].size());
    
    std::cout << "\n-> Iniciando entrenamiento SVM One-vs-All" << std::endl;
    std::cout << "   Muestras: " << m << ", Dimensiones: " << n << std::endl;
    std::cout << "   Kernel: Lineal" << std::endl;
    
    // Contar muestras por clase
    std::map<int, int> muestras_por_clase;
    for (int label : y) {
        muestras_por_clase[label]++;
    }
    
    // Extraer lista de clases
    std::vector<int> clases;
    for (const auto& [clase, _] : muestras_por_clase) {
        clases.push_back(clase);
    }
    
    // Inicializar modelo
    ModeloSVM modelo;
    modelo.clases = clases;
    modelo.dimensionCaracteristicas = n;
    
    // Mostrar configuracion
    std::cout << "\n-> Configuracion de entrenamiento:" << std::endl;
    std::cout << "   Optimizer: " << (cfg.usarAdamOptimizer ? "Adam" : "SGD+Momentum") << std::endl;
    std::cout << "   LR: " << cfg.tasaAprendizaje << " (decay exp: 0.9996)" << std::endl;
    std::cout << "   C: " << cfg.C << " | Momentum: " << cfg.momentum 
              << " | Batch: " << cfg.batchSizeNormal << std::endl;
    std::cout << "   Objetivo: Spec>=" << cfg.specificityTarget
              << "%, Recall>=" << cfg.recallMinimo 
              << "%, Prec>=" << cfg.precisionMinima
              << "%, F1>=" << cfg.f1Minimo << "%" << std::endl;
    std::cout << "   Peso max clase: " << cfg.pesoMaximo 
              << " (factor=" << cfg.factorPesoConservador << ")" << std::endl;
    
    // Crear directorio para curvas ROC si esta habilitado
    if (cfg.exportarROC) {
        std::cout << "\n-> Exportacion de curvas ROC: HABILITADO" << std::endl;
#ifdef _WIN32
        system("if not exist roc_data mkdir roc_data");
#else
        system("mkdir -p roc_data");
#endif
        std::cout << "   Directorio: roc_data/" << std::endl;
    }
    
    // ========================================================================
    // ENTRENAR UN CLASIFICADOR BINARIO POR CADA CLASE (PARALELO CON OPENMP)
    // ========================================================================
    
    int num_clases = static_cast<int>(clases.size());
    
    // Pre-asignar espacio para resultados (thread-safe)
    modelo.pesosPorClase.resize(num_clases);
    modelo.biasPorClase.resize(num_clases);
    modelo.plattAPorClase.resize(num_clases, 0.0);
    modelo.plattBPorClase.resize(num_clases, 0.0);
    modelo.thresholdsPorClase.resize(num_clases, 0.0);
    
#ifdef _OPENMP
    std::cout << "\n-> Modo PARALELO activado: " << omp_get_max_threads() 
              << " threads disponibles" << std::endl;
    std::cout << "   Se entrenaran multiples clases simultaneamente" << std::endl;
#else
    std::cout << "\n-> Modo SERIAL (OpenMP desactivado)" << std::endl;
#endif
    
    // LOOP PARALELIZABLE: cada clasificador es independiente
    OMP_PARALLEL_FOR
    for (int idx = 0; idx < num_clases; ++idx) {
        int clase = clases[idx];
        int muestras_esta_clase = muestras_por_clase[clase];
        
        // Log thread-safe (critical section)
        OMP_CRITICAL
        {
            std::cout << "\n-> [Thread " << obtenerThreadID() << "] Entrenando hablante " 
                      << clase << " (" << muestras_esta_clase << " muestras)" << std::endl;
        }
        
        // Preparar datos binarios (One-vs-All)
        int positivas, negativas;
        std::vector<int> y_binario = prepararDatosBinarios(y, clase, positivas, negativas);
        
        AudioSample ratio = static_cast<AudioSample>(negativas) / positivas;
        
        OMP_CRITICAL
        {
            std::cout << "   [Thread " << obtenerThreadID() << "] Clase " << clase 
                      << " | Distribucion: " << positivas << " pos, "
                      << negativas << " neg (ratio 1:"
                      << std::fixed << std::setprecision(1) << ratio << ")" << std::endl;
        }
        
        // ENTRENAR CLASIFICADOR BINARIO (FUNCION CORE)
        // Cada thread entrena su clasificador de forma independiente
        ResultadoEntrenamiento resultado = entrenarClasificadorBinario(
            X, y_binario, cfg, CONFIG_DATASET.seed + idx  // Seed diferente por thread
        );
        
        // Verificar si el entrenamiento fue exitoso
        if (!resultado.entrenamiento_exitoso) {
            OMP_CRITICAL
            {
                std::cerr << "   ! WARNING [Thread " << obtenerThreadID() 
                          << "]: Clase " << clase << " no convergio adecuadamente" << std::endl;
            }
        }
        
        // Imprimir metricas finales (thread-safe)
        OMP_CRITICAL
        {
            std::cout << "   [Thread " << obtenerThreadID() << "] Clase " << clase 
                      << " FINAL: Rec=" << std::fixed << std::setprecision(1)
                      << resultado.recall_final << "% Spe=" << resultado.specificity_final
                      << "% Pre=" << resultado.precision_final
                      << "% F1=" << resultado.f1_final
                      << "% | b=" << std::setprecision(3) << resultado.bias
                      << " (TP=" << resultado.tp << " FN=" << resultado.fn
                      << " TN=" << resultado.tn << " FP=" << resultado.fp << ")" << std::endl;
        }
        
        // Imprimir metricas biometricas robustas si esta habilitado
        if (cfg.imprimirMetricasRobustas && resultado.metricas_biometricas_validas) {
            OMP_CRITICAL
            {
                std::cout << "   [Thread " << obtenerThreadID() << "] Clase " << clase
                          << " BIOMETRICAS: FAR=" << std::fixed << std::setprecision(2)
                          << resultado.FAR << "% FRR=" << resultado.FRR
                          << "% EER=" << resultado.EER << "% | AUC="
                          << std::setprecision(4) << resultado.AUC
                          << " (threshold_EER=" << std::setprecision(3)
                          << resultado.threshold_eer << ")" << std::endl;
            }
        }
        
        // Exportar curva ROC si esta habilitado (thread-safe)
        if (cfg.exportarROC && !resultado.scores_finales.empty()) {
            CurvaROC curva = calcularCurvaROC(
                resultado.scores_finales,
                resultado.y_binario_final,
                200
            );
            
            std::string ruta_csv = "roc_data/roc_clase_" + std::to_string(clase) + ".csv";
            
            OMP_CRITICAL
            {
                exportarROC_CSV(curva, ruta_csv, clase);
            }
        }
        
        // Guardar clasificador en posicion pre-asignada (thread-safe por indice unico)
        modelo.pesosPorClase[idx] = resultado.pesos;
        modelo.biasPorClase[idx] = resultado.bias;
        modelo.plattAPorClase[idx] = 0.0;
        modelo.plattBPorClase[idx] = 0.0;
        modelo.thresholdsPorClase[idx] = 0.0;
    }
    
    std::cout << "\n-> Entrenamiento completado para " << clases.size()
              << " clases" << std::endl;
    
    return modelo;
}