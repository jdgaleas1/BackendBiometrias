#include "../../utils/config.h"
#include "../../core/classification/svm.h"
#include "../../core/process_dataset/dataset.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// SISTEMA DE PROFILING DE RENDIMIENTO

struct MetricasRendimiento
{
    double tiempoMs = 0.0;
    double ramPeakMB = 0.0;
    double ramPromMB = 0.0;
    double cpuProm = 0.0;
    int numMuestras = 0;
};

struct ProfilerEtapa
{
    std::string nombre;
    std::chrono::high_resolution_clock::time_point inicio;
    std::vector<double> muestrasRAM;
    std::vector<double> muestrasCPU;
    std::atomic<bool> activo{false};
    std::mutex mutexMuestras;

    ProfilerEtapa(const std::string &n) : nombre(n) {}

    void iniciar()
    {
        if (!CONFIG_PROFILING.habilitado)
            return;
        inicio = std::chrono::high_resolution_clock::now();
        activo = true;
        muestrasRAM.clear();
        muestrasCPU.clear();
    }

    void detener()
    {
        activo = false;
    }

    void agregarMuestra(double ramMB, double cpuPct)
    {
        std::lock_guard<std::mutex> lock(mutexMuestras);
        if (CONFIG_PROFILING.medirRAM)
            muestrasRAM.push_back(ramMB);
        if (CONFIG_PROFILING.medirCPU)
            muestrasCPU.push_back(cpuPct);
    }

    MetricasRendimiento obtenerMetricas()
    {
        MetricasRendimiento metricas;
        auto fin = std::chrono::high_resolution_clock::now();
        metricas.tiempoMs = std::chrono::duration<double, std::milli>(fin - inicio).count();

        if (!muestrasRAM.empty())
        {
            metricas.ramPeakMB = *std::max_element(muestrasRAM.begin(), muestrasRAM.end());
            metricas.ramPromMB = std::accumulate(muestrasRAM.begin(), muestrasRAM.end(), 0.0) / muestrasRAM.size();
        }

        if (!muestrasCPU.empty())
        {
            metricas.cpuProm = std::accumulate(muestrasCPU.begin(), muestrasCPU.end(), 0.0) / muestrasCPU.size();
        }

        metricas.numMuestras = static_cast<int>(muestrasRAM.size());
        return metricas;
    }
};

// Funciones auxiliares de profiling

inline double obtenerRAMUsadaMB()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
    {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        return usage.ru_maxrss / 1024.0;
    }
#endif
    return 0.0;
}

inline double obtenerCPUPorcentaje()
{
#ifdef _WIN32
    static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors = 0;
    static bool primera = true;

    if (primera)
    {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;

        FILETIME ftime, fsys, fuser;
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&lastCPU, &ftime, sizeof(FILETIME));

        GetProcessTimes(GetCurrentProcess(), &ftime, &ftime, &fsys, &fuser);
        memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
        memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));
        primera = false;
        return 0.0;
    }

    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;

    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));

    GetProcessTimes(GetCurrentProcess(), &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));

    double percent = (double)((sys.QuadPart - lastSysCPU.QuadPart) +
                              (user.QuadPart - lastUserCPU.QuadPart));
    percent /= (now.QuadPart - lastCPU.QuadPart);
    percent /= numProcessors;
    lastCPU = now;
    lastUserCPU = user;
    lastSysCPU = sys;

    return percent * 100.0;
#else
    return 0.0;
#endif
}

void monitorearRecursos(ProfilerEtapa *profiler)
{
    if (!CONFIG_PROFILING.habilitado)
        return;

    while (profiler->activo)
    {
        double ram = obtenerRAMUsadaMB();
        double cpu = obtenerCPUPorcentaje();
        profiler->agregarMuestra(ram, cpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(CONFIG_PROFILING.intervaloMuestreoMs));
    }
}

static bool validarCompatibilidad(const std::vector<std::vector<AudioSample>> &X_train,
                                  const std::vector<std::vector<AudioSample>> &X_test)
{
    if (X_train.empty() || X_test.empty())
    {
        std::cerr << "! Error: Conjuntos de datos vacios" << std::endl;
        return false;
    }

    if (X_train[0].size() != X_test[0].size())
    {
        std::cerr << "! ERROR CRITICO: Dimensiones inconsistentes!\n"
                  << "  Train: " << X_train[0].size() << " caracteristicas\n"
                  << "  Test: " << X_test[0].size() << " caracteristicas\n"
                  << "  Regenera ambos datasets con la misma configuracion." << std::endl;
        return false;
    }

    return true;
}

// MAIN
int main(int argc, char *argv[])
{
    std::cout << std::string(70, '*') << std::endl;
    std::cout << "*  ENTRENAMIENTO SVM - SISTEMA BIOMETRICO DE VOZ  *" << std::endl;
    std::cout << std::string(70, '*') << std::endl;

    // CONFIGURACION DE RUTAS
    std::string rutaTrain = obtenerRutaDatasetTrain();
    std::string rutaTest = obtenerRutaDatasetTest();
    std::string rutaModelo = obtenerRutaModelo();

    // Crear directorio para el modelo
    fs::create_directories("model");

    // Procesar argumentos opcionales
    if (argc >= 2)
        rutaTrain = argv[1];
    if (argc >= 3)
        rutaTest = argv[2];
    if (argc >= 4)
        rutaModelo = argv[3];

    // CARGA DE DATASETS
    std::vector<std::vector<AudioSample>> X_train, X_test;
    std::vector<int> y_train, y_test;

    ProfilerEtapa profilerCarga("Carga_Datasets");
    std::thread monitorThreadCarga;

    if (CONFIG_PROFILING.habilitado)
    {
        profilerCarga.iniciar();
        monitorThreadCarga = std::thread(monitorearRecursos, &profilerCarga);
    }

    std::cout << "\n-> Cargando datos de entrenamiento..." << std::endl;
    if (!cargarDatasetBinario(rutaTrain, X_train, y_train))
    {
        std::cerr << "! Error al cargar datos de entrenamiento" << std::endl;
        return -1;
    }

    std::cout << "\n-> Cargando datos de prueba..." << std::endl;
    if (!cargarDatasetBinario(rutaTest, X_test, y_test))
    {
        std::cerr << "! Error al cargar datos de prueba" << std::endl;
        return -1;
    }

    MetricasRendimiento metricasCarga;
    if (CONFIG_PROFILING.habilitado)
    {
        profilerCarga.detener();
        if (monitorThreadCarga.joinable())
            monitorThreadCarga.join();
        metricasCarga = profilerCarga.obtenerMetricas();
    }

    // Validar compatibilidad
    if (!validarCompatibilidad(X_train, X_test))
    {
        return -1;
    }

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    if (CONFIG_SVM.usarExpansionPolinomial)
    {
        std::cout << "EXPANSION POLINOMIAL: ACTIVADA (ya aplicada en dataset)" << std::endl;
    }
    else
    {
        std::cout << "EXPANSION POLINOMIAL: DESACTIVADA" << std::endl;
    }
    std::cout << "Dimension del dataset: " << X_train[0].size() << " features" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    if (CONFIG_SVM.usarNormalizacionL2)
    {
        std::cout << "NORMALIZACION L2: ACTIVADA (aplicada en pipeline)" << std::endl;
    }
    else
    {
        std::cout << "NORMALIZACION L2: DESACTIVADA" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    // DIAGNOSTICO DEL DATASET
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "AUGMENTATION DATA" << std::endl;
    if (CONFIG_DATASET.usarAugmentation && CONFIG_AUG.numVariaciones > 0)
    {
        std::cout << "AUGMENTATION: ACTIVADO" << std::endl;
        std::cout << "Variaciones por audio: " << (CONFIG_AUG.numVariaciones + 1)
                  << " (1 original + " << CONFIG_AUG.numVariaciones << " perturbadas)" << std::endl;
    }
    else
    {
        std::cout << "AUGMENTATION: DESACTIVADO (sin variaciones)" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "DIAGNOSTICO DEL DATASET" << std::endl;

    std::cout << "\n-> Conjunto de ENTRENAMIENTO:" << std::endl;
    diagnosticarDataset(X_train, y_train);

    std::cout << "\n-> Conjunto de PRUEBA:" << std::endl;
    diagnosticarDataset(X_test, y_test);

    std::cout << std::string(70, '=') << std::endl;

    // MOSTRAR CONFIGURACION
    std::cout << "\n-> Configuracion de entrenamiento (de config.h):" << std::endl;
    CONFIG_SVM.mostrar();

    // ENTRENAMIENTO
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "INICIANDO ENTRENAMIENTO" << std::endl;

    ProfilerEtapa profilerEntrenamiento("Entrenamiento_SVM");
    std::thread monitorThreadEntrenamiento;

    if (CONFIG_PROFILING.habilitado)
    {
        profilerEntrenamiento.iniciar();
        monitorThreadEntrenamiento = std::thread(monitorearRecursos, &profilerEntrenamiento);
    }

    ModeloSVM modelo = entrenarSVMOVA(X_train, y_train);

    MetricasRendimiento metricasEntrenamiento;
    if (CONFIG_PROFILING.habilitado)
    {
        profilerEntrenamiento.detener();
        if (monitorThreadEntrenamiento.joinable())
            monitorThreadEntrenamiento.join();
        metricasEntrenamiento = profilerEntrenamiento.obtenerMetricas();
    }

    // GUARDAR MODELO
    std::cout << "\n-> Guardando modelo entrenado en..." << std::endl;
    if (guardarModeloModular(rutaModelo, modelo))
    {
        std::cout << "   & Modelo guardado en: " << rutaModelo << std::endl;
    }
    else
    {
        std::cerr << "! Error al guardar el modelo " << std::endl;
        return -1;
    }

    // EVALUACION DEL MODELO
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "EVALUACION DEL MODELO" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    ProfilerEtapa profilerEvaluacion("Evaluacion_Modelo");
    std::thread monitorThreadEvaluacion;

    if (CONFIG_PROFILING.habilitado)
    {
        profilerEvaluacion.iniciar();
        monitorThreadEvaluacion = std::thread(monitorearRecursos, &profilerEvaluacion);
    }

    evaluarModeloCompleto(X_train, y_train, modelo, "ENTRENAMIENTO");
    evaluarModeloCompleto(X_test, y_test, modelo, "PRUEBA");

    MetricasRendimiento metricasEvaluacion;
    if (CONFIG_PROFILING.habilitado)
    {
        profilerEvaluacion.detener();
        if (monitorThreadEvaluacion.joinable())
            monitorThreadEvaluacion.join();
        metricasEvaluacion = profilerEvaluacion.obtenerMetricas();
    }

    // RESUMEN DE PROFILING =======================================================
    if (CONFIG_PROFILING.habilitado)
    {
        std::cout << "\n"
                  << std::string(70, '*') << std::endl;
        std::cout << "*  RESUMEN DE PROFILING DE RENDIMIENTO  *" << std::endl;
        std::cout << std::string(70, '*') << std::endl;

        std::cout << "\n# CARGA DE DATASETS:" << std::endl;
        if (CONFIG_PROFILING.medirTiempo)
        {
            std::cout << "   Tiempo: " << std::fixed << std::setprecision(2)
                      << (metricasCarga.tiempoMs / 1000.0) << " segundos" << std::endl;
        }
        if (CONFIG_PROFILING.medirRAM)
        {
            std::cout << "   RAM Peak: " << std::fixed << std::setprecision(1)
                      << metricasCarga.ramPeakMB << " MB" << std::endl;
            std::cout << "   RAM Promedio: " << std::fixed << std::setprecision(1)
                      << metricasCarga.ramPromMB << " MB" << std::endl;
        }
        if (CONFIG_PROFILING.medirCPU)
        {
            std::cout << "   CPU Promedio: " << std::fixed << std::setprecision(1)
                      << metricasCarga.cpuProm << " %" << std::endl;
        }

        std::cout << "\n# ENTRENAMIENTO SVM:" << std::endl;
        if (CONFIG_PROFILING.medirTiempo)
        {
            std::cout << "   Tiempo: " << std::fixed << std::setprecision(2)
                      << (metricasEntrenamiento.tiempoMs / 1000.0) << " segundos" << std::endl;
        }
        if (CONFIG_PROFILING.medirRAM)
        {
            std::cout << "   RAM Peak: " << std::fixed << std::setprecision(1)
                      << metricasEntrenamiento.ramPeakMB << " MB" << std::endl;
            std::cout << "   RAM Promedio: " << std::fixed << std::setprecision(1)
                      << metricasEntrenamiento.ramPromMB << " MB" << std::endl;
        }
        if (CONFIG_PROFILING.medirCPU)
        {
            std::cout << "   CPU Promedio: " << std::fixed << std::setprecision(1)
                      << metricasEntrenamiento.cpuProm << " %" << std::endl;
        }

        std::cout << "\n# EVALUACION DEL MODELO:" << std::endl;
        if (CONFIG_PROFILING.medirTiempo)
        {
            std::cout << "   Tiempo: " << std::fixed << std::setprecision(2)
                      << (metricasEvaluacion.tiempoMs / 1000.0) << " segundos" << std::endl;
        }
        if (CONFIG_PROFILING.medirRAM)
        {
            std::cout << "   RAM Peak: " << std::fixed << std::setprecision(1)
                      << metricasEvaluacion.ramPeakMB << " MB" << std::endl;
            std::cout << "   RAM Promedio: " << std::fixed << std::setprecision(1)
                      << metricasEvaluacion.ramPromMB << " MB" << std::endl;
        }
        if (CONFIG_PROFILING.medirCPU)
        {
            std::cout << "   CPU Promedio: " << std::fixed << std::setprecision(1)
                      << metricasEvaluacion.cpuProm << " %" << std::endl;
        }

        double tiempoTotalS = (metricasCarga.tiempoMs + metricasEntrenamiento.tiempoMs +
                               metricasEvaluacion.tiempoMs) /
                              1000.0;
        double ramMaxTotal = std::max({metricasCarga.ramPeakMB,
                                       metricasEntrenamiento.ramPeakMB,
                                       metricasEvaluacion.ramPeakMB});

        std::cout << "\n# TOTALES:" << std::endl;
        std::cout << "   Tiempo total: " << std::fixed << std::setprecision(2)
                  << tiempoTotalS << " segundos" << std::endl;
        std::cout << "   RAM Peak global: " << std::fixed << std::setprecision(1)
                  << ramMaxTotal << " MB" << std::endl;
        std::cout << "   Samples entrenamiento: " << X_train.size() << std::endl;
        std::cout << "   Samples evaluacion: " << (X_train.size() + X_test.size()) << std::endl;

        std::cout << "\n"
                  << std::string(70, '*') << std::endl;
    }

    return 0;
}