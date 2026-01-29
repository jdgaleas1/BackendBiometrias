// config.h - VERSION 3.0 REFACTORIZADO
// Cambios: float -> double, control OpenMP, mejores practicas
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <utility>

// CONTROL DE PRECISION Y PARALELISMO

// Tipo unificado para muestras de audio (double para precision biometrica)
using AudioSample = double;

// CONTROL AVANZADO DE PARALELIZACION CON OPENMP
// Se detecta automaticamente si el compilador tiene OpenMP habilitado (_OPENMP)
// Define ENABLE_OPENMP=0 para forzar modo serial incluso con OpenMP disponible
//
// IMPORTANTE: Cambiar este valor requiere recompilar completamente
// Para cambiar en tiempo de compilacion: cmake -DENABLE_OPENMP=OFF
#ifndef ENABLE_OPENMP
    #ifdef _OPENMP
        #define ENABLE_OPENMP 1 // Detectado: compilador con OpenMP
    #else
        #define ENABLE_OPENMP 0 // Sin OpenMP: modo serial
    #endif
#endif

#if ENABLE_OPENMP
#include <omp.h>
#define OMP_ENABLED true

// Macros utiles para pragmas condicionales
#define OMP_PRAGMA(directive) _Pragma(#directive)
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_CRITICAL _Pragma("omp critical")
#define OMP_REDUCTION(op, var) _Pragma("omp parallel for reduction(" #op ":" #var ")")

// Funcion inline para obtener numero de threads
inline int obtenerNumThreads()
{
    return omp_get_max_threads();
}

// Funcion inline para obtener ID del thread actual
inline int obtenerThreadID()
{
    return omp_get_thread_num();
}

#else
// Modo SERIAL: todas las macros se convierten en NO-OP
#define OMP_ENABLED false
#define OMP_PRAGMA(directive)
#define OMP_PARALLEL_FOR
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_CRITICAL
#define OMP_REDUCTION(op, var)

// Funciones dummy para modo serial
inline int obtenerNumThreads() { return 1; }
inline int obtenerThreadID() { return 0; }

#endif

// Utilidad para imprimir estado de OpenMP
inline void mostrarEstadoOpenMP()
{
    std::cout << "-> OpenMP: " << (OMP_ENABLED ? "ACTIVADO" : "DESACTIVADO");
#ifdef _OPENMP
    std::cout << " (" << omp_get_max_threads() << " threads disponibles)";
#endif
    std::cout << std::endl;
}

// FUNCIONES DE RUTA

inline std::string obtenerRutaBase()
{
#ifdef _WIN32
    return "";
#else
    const char *appDir = std::getenv("APP_DIR");
    return appDir ? std::string(appDir) + "/" : "/app/";
#endif
}

inline std::string obtenerRutaModelo()
{
    return obtenerRutaBase() + "model/";
}
inline std::string obtenerRutaMapping()
{
    return obtenerRutaBase() + "model/metadata.json";
}
inline std::string obtenerRutaDatasetTrain()
{
    return obtenerRutaBase() + "processed_dataset_bin/caracteristicas_train.dat";
}

inline std::string obtenerRutaDatasetTest()
{
    return obtenerRutaBase() + "processed_dataset_bin/caracteristicas_test.dat";
}

inline std::string obtenerRutaTempAudio()
{
    return obtenerRutaBase() + "temp_audio/";
}

// CONFIGURACION POSTGREST
// Retorna host y puerto directamente para httplib::Client(host, port)
inline std::pair<std::string, int> obtenerPostgRESTConfig()
{
    const char* postgrestUrl = std::getenv("POSTGREST_URL");
    
    if (postgrestUrl != nullptr) {
        // En Docker: POSTGREST_URL=http://biometria_api:3000
        std::string url(postgrestUrl);
        
        // Parseo mejorado: extraer host:puerto de http://host:puerto
        size_t hostStart = url.find("://");
        if (hostStart != std::string::npos) {
            hostStart += 3; // saltar "://"
            
            // Buscar el final del host (puede ser ':' para puerto o fin de string)
            size_t hostEnd = url.find('/', hostStart);  // Buscar '/' si existe path
            if (hostEnd == std::string::npos) {
                hostEnd = url.length();  // No hay path, tomar hasta el final
            }
            
            std::string hostPort = url.substr(hostStart, hostEnd - hostStart);
            
            // Ahora separar host:puerto
            size_t colonPos = hostPort.find(':');
            if (colonPos != std::string::npos) {
                std::string host = hostPort.substr(0, colonPos);
                int port = std::stoi(hostPort.substr(colonPos + 1));
                return {host, port};
            } else {
                // Sin puerto explícito, asumir 3000
                return {hostPort, 3000};
            }
        }
    }
    
    // Desarrollo local: localhost:3001
    return {"localhost", 3001};
}

// CONFIG AUGMENTATION
struct ConfigAugmentation
{
    double intensidadRuido;
    double volumenMin;
    double volumenMax;
    double velocidadMin;
    double velocidadMax;
    int numVariaciones;
    bool verbose;
    unsigned int seed;

    ConfigAugmentation()
    {
        intensidadRuido = 0.05;
        volumenMin = 0.70;
        volumenMax = 1.30;
        velocidadMin = 0.85;
        velocidadMax = 1.15;
        numVariaciones = 0;  // 5 muestras: 1 original + 4 variaciones
        verbose = false;
        seed = 42;
    }

    void mostrar() const
    {
        std::cout << "-> Config Augmentation:" << std::endl;
        std::cout << "   Intensidad ruido: " << intensidadRuido << std::endl;
        std::cout << "   Volumen: [" << volumenMin << ", " << volumenMax << "]" << std::endl;
        std::cout << "   Velocidad: [" << velocidadMin << ", " << velocidadMax << "]" << std::endl;
        std::cout << "   Variaciones: " << (numVariaciones + 1) << " (1 original + "
                  << numVariaciones << " perturbadas)" << std::endl;
    }
};

// CONFIG PREPROCESSING
struct ConfigPreprocessing
{
    // SECCION 1: VAD (Voice Activity Detection)
    double vadEnergyThreshold; // Umbral de energia minima
    int vadMinDurationMs;      // Duracion minima de segmento (ms)
    int vadPaddingMs;          // Padding antes/despues de voz (ms)
    int vadFrameSizeMs;        // Tamaño de frame para analisis (ms)
    int vadFrameStrideMs;      // Stride entre frames (ms)
    int vadMergeGapMs;         // Gap maximo para unir segmentos (ms)

    // SECCION 2: NORMALIZACION
    double normalizationTargetRMS; // RMS objetivo (0.05-0.2, default 0.1)

    // SECCION 4: CONTROL GLOBAL
    bool enablePreprocessing; // Master switch (false = bypass total)
    bool verbose;             // Logging detallado

    ConfigPreprocessing()
    {
        // === VAD optimizado ===
        vadEnergyThreshold = 0.0005;
        vadMinDurationMs = 100;
        vadPaddingMs = 150;
        vadFrameSizeMs = 25;
        vadFrameStrideMs = 10;
        vadMergeGapMs = 250;
        // === NORMALIZACION ===
        normalizationTargetRMS = 0.1;

        // === CONTROL GLOBAL ===
        enablePreprocessing = true;
        verbose = true;
    }

    void mostrar() const
    {
        std::cout << "   OpenMP: " << (OMP_ENABLED ? "ACTIVADO" : "DESACTIVADO");
#ifdef _OPENMP
        std::cout << " (threads max=" << omp_get_max_threads() << ")";
#endif
        std::cout << std::endl;

        if (!enablePreprocessing)
        {
            std::cout << "   *** PREPROCESAMIENTO DESHABILITADO (BYPASS) ***" << std::endl;
            return;
        }

        // VAD
        std::cout << "\n   [VAD]" << std::endl;
        std::cout << "     Energy threshold: " << vadEnergyThreshold << std::endl;
        std::cout << "     Frame: " << vadFrameSizeMs << "ms (stride " << vadFrameStrideMs << "ms)" << std::endl;
        std::cout << "     Padding: " << vadPaddingMs << "ms | Gap merge: " << vadMergeGapMs << "ms" << std::endl;

        // NORMALIZACION
        std::cout << "\n   [NORMALIZACION]" << std::endl;
        std::cout << "     Target RMS: " << normalizationTargetRMS << std::endl;
    }
};

// CONFIG MFCC
struct ConfigMFCC
{
    int numCoefficients;
    int numFilters;
    double freqMin;
    double freqMax;
    int totalFeatures;

    ConfigMFCC()
    {
        numCoefficients = 50;
        numFilters = 40;
        freqMin = 0.0;
        freqMax = 8000.0;
        totalFeatures = 250;  // 5 estadisticas x 50 coefs = 250 features
    }

    void mostrar() const
    {
        std::cout << "-> Config MFCC:" << std::endl;
        std::cout << "   Coeficientes: " << numCoefficients << std::endl;
        std::cout << "   Features totales: " << totalFeatures << " (MEAN+STD+MIN+MAX+DELTA de " << numCoefficients << " coefs)" << std::endl;
        std::cout << "   Filtros mel: " << numFilters << std::endl;
        std::cout << "   Rango frecuencia: [" << freqMin << ", " << freqMax << "] Hz" << std::endl;
    }
};

// CONFIG STFT
struct ConfigSTFT
{
    int frameSizeMs;
    int frameStrideMs;

    ConfigSTFT()
    {
        frameSizeMs = 25;
        frameStrideMs = 10;
    }

    void mostrar() const
    {
        std::cout << "-> Config STFT:" << std::endl;
        std::cout << "   Frame size: " << frameSizeMs << " ms" << std::endl;
        std::cout << "   Frame stride: " << frameStrideMs << " ms" << std::endl;
    }
};

// CONFIG SVM
struct ConfigSVM
{
    double tasaAprendizaje;
    int epocas;
    double C;
    bool usarNormalizacionL2;
    bool usarExpansionPolinomial;  // Expandir features a grado 2 (50->100 o 250->500)

    // Configuraciones adicionales
    double momentum;
    double specificityTarget;
    double recallMinimo;
    double precisionMinima;
    double f1Minimo;
    int epocasMinimas;

    // Pesos para balanceo
    bool usarPesoLogaritmico;
    double factorPesoConservador;
    double pesoMinimo;
    double pesoMaximo;
    double umbralRecallColapso;

    // Adam optimizer
    bool usarAdamOptimizer;
    double beta1Adam;
    double beta2Adam;
    double epsilonAdam;

    // Paciencia
    int paciencia;
    int pacienciaMinoritaria;
    int batchSizeNormal;
    int muestrasMinoritarias;
    int seed;

    bool exportarROC;         // Exportar curvas ROC a CSV
    bool imprimirMetricasRobustas; // Imprimir metricas robustas

    ConfigSVM()
    {
        // OPTIMIZADO PARA 67+ CLASES CON MODELO LINEAL
        tasaAprendizaje = 0.005;     // Reducido para mayor estabilidad
        epocas = 40000;              // Aumentado para convergencia
        C = 10.0;                    // CRITICO: Regularizacion fuerte (67+ clases)
        usarNormalizacionL2 = true;  
        usarExpansionPolinomial = false;  // DESACTIVADA: causa overfitting con muchas clases

        momentum = 0.9;
        specificityTarget = 88.0;    // Aumentado para mayor precision
        recallMinimo = 75.0;         // Aumentado para reducir falsos negativos
        precisionMinima = 75.0;      // Aumentado
        f1Minimo = 75.0;             // Aumentado
        epocasMinimas = 800;         // Mas epocas minimas

        usarPesoLogaritmico = false;
        factorPesoConservador = 1.5; // Aumentado para clases minoritarias
        pesoMinimo = 1.0;
        pesoMaximo = 15.0;           // Aumentado para mejor balance
        umbralRecallColapso = 98.0;

        usarAdamOptimizer = true;
        beta1Adam = 0.9;
        beta2Adam = 0.999;
        epsilonAdam = 1e-8;

        paciencia = 1500;            // Mayor paciencia
        pacienciaMinoritaria = 2000;
        batchSizeNormal = 32;
        muestrasMinoritarias = 50;
        seed = 42;

        imprimirMetricasRobustas = true;
        exportarROC = false;  // Desactivado para optimizar entrenamiento      

    }

    void mostrar() const
    {
        std::cout << "-> Config SVM:" << std::endl;
        std::cout << "   Modo lineal "

                  << std::endl;
        std::cout << "   C: " << C << std::endl;

        std::cout << "   Tasa aprendizaje: " << tasaAprendizaje << std::endl;
        std::cout << "   Epocas: " << epocas << std::endl;
        std::cout << "   Optimizer: "
                  << (usarAdamOptimizer ? "Adam" : "SGD+Momentum")
                  << std::endl;

        std::cout << "   Normalizacion L2: "
                  << (usarNormalizacionL2 ? "SI" : "NO") << std::endl;
        std::cout << "   Expansion Polinomial: "
                  << (usarExpansionPolinomial ? "SI (dimension x2)" : "NO") << std::endl;
    }
};

// CONFIG AUTENTICACION
struct ConfigAutenticacion
{
    // Umbrales para control de acceso biometrico
    double scoreMinimo;          // Score minimo absoluto para autenticar
    double diferenciaMinima;     // Separacion minima del segundo lugar
    double factorSegundoLugar;   // El segundo debe ser < (maxScore * factor)
    double umbralScoreAlto;      // Score considerado "muy alto"
    
    ConfigAutenticacion()
    {
        // UMBRALES AJUSTADOS PARA 67+ CLASES - MODELO LINEAL
        // Con 250 features (sin expansion polinomial):
        //  - Usuarios correctos:  -0.3 a +1.2 (típico: 0.4 - 0.9)
        //  - Impostores cercanos: -1.0 a +0.3
        //  - Impostores lejanos:  -5.0 a -1.5
        
        scoreMinimo = 0.1;           // Score minimo mas estricto (67+ clases)
        diferenciaMinima = 0.20;     // Mayor separacion requerida
        factorSegundoLugar = 0.75;   // Segundo debe ser < 75% del primero (mas estricto)
        umbralScoreAlto = 0.8;       // Score considerado muy confiable
    }
    
    void mostrar() const
    {
        std::cout << "-> Config Autenticacion:" << std::endl;
        std::cout << "   Score minimo: " << scoreMinimo << std::endl;
        std::cout << "   Diferencia minima: " << diferenciaMinima << std::endl;
        std::cout << "   Factor segundo lugar: " << factorSegundoLugar << std::endl;
        std::cout << "   Umbral score alto: " << umbralScoreAlto << std::endl;
    }
};

// CONFIG DATASET
struct ConfigDataset
{
    double trainRatio;
    int minAudioSamples;
    bool usarAugmentation;
    int seed;
    int minAudiosPorHablante;
    bool usarDivisionManual;
    int muestrasTrainPorHablante;
    int muestrasTestPorHablante;

    ConfigDataset()
    {
        trainRatio = 0.8;
        minAudioSamples = 3048;
        usarAugmentation = true;
        seed = 42;
        minAudiosPorHablante = 7;
        usarDivisionManual = true;
        muestrasTrainPorHablante = 6;
        muestrasTestPorHablante = 1;
    }

    void mostrar() const
    {
        std::cout << "-> Config Dataset:" << std::endl;
        if (usarDivisionManual)
        {
            std::cout << "   Division MANUAL: " << muestrasTrainPorHablante
                      << " train + " << muestrasTestPorHablante << " test" << std::endl;
        }
        else
        {
            std::cout << "   Train/Test: " << (trainRatio * 100) << "/"
                      << ((1 - trainRatio) * 100) << "%" << std::endl;
        }
        std::cout << "   Augmentation: " << (usarAugmentation ? "SI" : "NO") << std::endl;
    }
};

// CONFIG PROFILING
struct ConfigProfiling
{
    bool habilitado;              // Master switch para profiling
    bool medirRAM;                // Medir consumo de RAM
    bool medirCPU;                // Medir uso de CPU
    bool medirTiempo;             // Medir tiempo por etapa
    int intervaloMuestreoMs;      // Intervalo de muestreo de metricas (ms)

    ConfigProfiling()
    {
        habilitado = true;
        medirRAM = true;
        medirCPU = true;
        medirTiempo = true;
        intervaloMuestreoMs = 100;
    }

    void mostrar() const
    {
        std::cout << "-> Config Profiling:" << std::endl;
        std::cout << "   Habilitado: " << (habilitado ? "SI" : "NO") << std::endl;
        if (habilitado)
        {
            std::cout << "   Medir RAM: " << (medirRAM ? "SI" : "NO") << std::endl;
            std::cout << "   Medir CPU: " << (medirCPU ? "SI" : "NO") << std::endl;
            std::cout << "   Medir Tiempo: " << (medirTiempo ? "SI" : "NO") << std::endl;
            std::cout << "   Intervalo muestreo: " << intervaloMuestreoMs << " ms" << std::endl;
        }
    }
};

// SINGLETON GLOBAL
class ConfigGlobal
{
public:
    ConfigAugmentation augmentation;
    ConfigPreprocessing preprocessing;
    ConfigMFCC mfcc;
    ConfigSTFT stft;
    ConfigSVM svm;
    ConfigAutenticacion autenticacion;
    ConfigDataset dataset;
    ConfigProfiling profiling;

    static ConfigGlobal &getInstance()
    {
        static ConfigGlobal instance;
        return instance;
    }

    void mostrarTodo() const
    {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "CONFIGURACION GLOBAL DEL SISTEMA v3.0" << std::endl;
        std::cout << "Precision: double (64-bit) | OpenMP: "
                  << (OMP_ENABLED ? "ACTIVADO" : "DESACTIVADO") << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        augmentation.mostrar();
        std::cout << std::endl;
        preprocessing.mostrar();
        std::cout << std::endl;
        mfcc.mostrar();
        std::cout << std::endl;
        stft.mostrar();
        std::cout << std::endl;
        svm.mostrar();
        std::cout << std::endl;
        autenticacion.mostrar();
        std::cout << std::endl;
        dataset.mostrar();
        std::cout << std::endl;
        profiling.mostrar();

        std::cout << std::string(60, '=') << std::endl;
    }

    ConfigGlobal(const ConfigGlobal &) = delete;
    ConfigGlobal &operator=(const ConfigGlobal &) = delete;

private:
    ConfigGlobal() = default;
};

// Macros de acceso global
#define CONFIG ConfigGlobal::getInstance()
#define CONFIG_AUG ConfigGlobal::getInstance().augmentation
#define CONFIG_PREP ConfigGlobal::getInstance().preprocessing
#define CONFIG_MFCC ConfigGlobal::getInstance().mfcc
#define CONFIG_STFT ConfigGlobal::getInstance().stft
#define CONFIG_SVM ConfigGlobal::getInstance().svm
#define CONFIG_AUTH ConfigGlobal::getInstance().autenticacion
#define CONFIG_DATASET ConfigGlobal::getInstance().dataset
#define CONFIG_PROFILING ConfigGlobal::getInstance().profiling

#endif // CONFIG_H
