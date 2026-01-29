#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <atomic>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>
#include <omp.h>
#include "../../utils/config.h"
#include "../../core/process_dataset/dataset.h"
#include "../../core/pipeline/audio_pipeline.h"
#include <numeric>

#ifdef _WIN32
#define NOMINMAX  // Evitar conflicto con std::min/std::max
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// SISTEMA DE PROFILING DE RENDIMIENTO

struct MetricasRendimiento {
    double tiempoMs = 0.0;
    double ramPeakMB = 0.0;
    double ramPromMB = 0.0;
    double cpuProm = 0.0;
    int numMuestras = 0;
};

struct ProfilerEtapa {
    std::string nombre;
    std::chrono::high_resolution_clock::time_point inicio;
    std::vector<double> muestrasRAM;
    std::vector<double> muestrasCPU;
    std::atomic<bool> activo{false};
    std::mutex mutexMuestras;

    ProfilerEtapa(const std::string& n) : nombre(n) {}

    void iniciar() {
        if (!CONFIG_PROFILING.habilitado) return;
        inicio = std::chrono::high_resolution_clock::now();
        activo = true;
        muestrasRAM.clear();
        muestrasCPU.clear();
    }

    void detener() {
        activo = false;
    }

    void agregarMuestra(double ramMB, double cpuPct) {
        std::lock_guard<std::mutex> lock(mutexMuestras);
        if (CONFIG_PROFILING.medirRAM) muestrasRAM.push_back(ramMB);
        if (CONFIG_PROFILING.medirCPU) muestrasCPU.push_back(cpuPct);
    }

    MetricasRendimiento obtenerMetricas() {
        MetricasRendimiento metricas;
        auto fin = std::chrono::high_resolution_clock::now();
        metricas.tiempoMs = std::chrono::duration<double, std::milli>(fin - inicio).count();

        if (!muestrasRAM.empty()) {
            metricas.ramPeakMB = *std::max_element(muestrasRAM.begin(), muestrasRAM.end());
            metricas.ramPromMB = std::accumulate(muestrasRAM.begin(), muestrasRAM.end(), 0.0) / muestrasRAM.size();
        }

        if (!muestrasCPU.empty()) {
            metricas.cpuProm = std::accumulate(muestrasCPU.begin(), muestrasCPU.end(), 0.0) / muestrasCPU.size();
        }

        metricas.numMuestras = static_cast<int>(muestrasRAM.size());
        return metricas;
    }
};

// Funciones auxiliares de profiling

inline double obtenerRAMUsadaMB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0; // Linux: KB -> MB
    }
#endif
    return 0.0;
}

inline double obtenerCPUPorcentaje() {
    // Aproximacion simple - en produccion usar herramientas mas precisas
#ifdef _WIN32
    static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors = 0;
    static bool primera = true;

    if (primera) {
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
    return 0.0; // TODO: implementar para Linux
#endif
}

void monitorearRecursos(ProfilerEtapa* profiler) {
    if (!CONFIG_PROFILING.habilitado) return;

    while (profiler->activo) {
        double ram = obtenerRAMUsadaMB();
        double cpu = obtenerCPUPorcentaje();
        profiler->agregarMuestra(ram, cpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(CONFIG_PROFILING.intervaloMuestreoMs));
    }
}

/**
 * Estructura para muestra de audio procesada
 */
struct AudioSampleData {
    fs::path path;
    int etiqueta;
    std::vector<AudioSample> features;
    bool procesado;

    AudioSampleData(const fs::path& p, int e)
        : path(p), etiqueta(e), procesado(false) {
        features.reserve(CONFIG_MFCC.totalFeatures);
    }

    AudioSampleData(const fs::path& p, int e, std::vector<AudioSample>&& f)
        : path(p), etiqueta(e), features(std::move(f)), procesado(true) {
    }
};

/**
 * Estadísticas thread-safe del procesamiento
 */
struct EstadisticasProcesamiento {
    std::atomic<int> totalProcesados{ 0 };
    std::atomic<int> totalExitosos{ 0 };
    std::atomic<int> totalFallidos{ 0 };
    std::atomic<int> audiosCrudos{ 0 };
    std::atomic<int> muestrasGeneradas{ 0 };
    std::map<int, int> muestrasPorHablante;

    void mostrarProgreso(int actual, int total, const std::string& archivo) {
        float progreso = (float)actual / total * 100.0f;
        std::cout << "-> [" << actual << "/" << total << " - "
            << std::fixed << std::setprecision(1) << progreso << "%] "
            << archivo << std::endl;
    }
};

// RECOPILACION DE ARCHIVOS
std::map<int, std::vector<fs::path>> recopilarArchivosPorHablante(
    const std::string& datasetPath,
    std::map<int, std::string>& idANombre) {

    std::map<int, std::vector<fs::path>> archivos;
    idANombre.clear();

    std::cout << "\n-> Recopilando archivos de audio..." << std::endl;
    std::cout << "   Dataset: " << datasetPath << std::endl;

    if (!fs::exists(datasetPath)) {
        std::cerr << "! Error: Directorio no existe: " << datasetPath << std::endl;
        return archivos;
    }

    std::set<std::string> extValidas = { ".mp3", ".wav", ".flac", ".aiff" };

    // Mapeo de nombre -> ID para asignar IDs automáticamente
    std::map<std::string, int> nombreAId;
    int siguienteId = 1;

    for (const auto& hablanteDir : fs::directory_iterator(datasetPath)) {
        if (!hablanteDir.is_directory()) continue;

        std::string nombreHablante = hablanteDir.path().filename().string();
        int idHablante;

        // Intentar convertir a número (datasets con IDs)
        bool esIdNumerico = false;
        try {
            idHablante = std::stoi(nombreHablante);
            esIdNumerico = true;
        }
        catch (...) {
            // Si no es número, asignar ID automáticamente
            if (nombreAId.find(nombreHablante) == nombreAId.end()) {
                nombreAId[nombreHablante] = siguienteId++;
                std::cout << "   * Asignando ID " << siguienteId - 1 << " a: " << nombreHablante << std::endl;
            }
            idHablante = nombreAId[nombreHablante];
            esIdNumerico = false;
        }

        // Guardar mapeo ID -> Nombre
        if (esIdNumerico) {
            idANombre[idHablante] = nombreHablante;  // ID como nombre
        } else {
            idANombre[idHablante] = nombreHablante;  // Nombre descriptivo
        }

        for (const auto& archivo : fs::directory_iterator(hablanteDir.path())) {
            if (!archivo.is_regular_file()) continue;

            std::string ext = archivo.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (extValidas.count(ext)) {
                archivos[idHablante].push_back(archivo.path());
            }
        }
    }

    // Filtrar hablantes con pocos audios
    for (auto it = archivos.begin(); it != archivos.end();) {
        if (it->second.size() < static_cast<size_t>(CONFIG_DATASET.minAudiosPorHablante)) {
            std::cerr << "% Warning: Hablante " << it->first
                << " descartado (solo " << it->second.size() << " audios)"
                << std::endl;
            it = archivos.erase(it);
        }
        else {
            ++it;
        }
    }

    std::cout << "   & Hablantes validos: " << archivos.size() << std::endl;
    return archivos;
}

// DIVISION TRAIN/TEST

void dividirTrainTest(
    const std::map<int, std::vector<fs::path>>& archivosPorHablante,
    std::vector<AudioSampleData>& trainSamples,
    std::vector<AudioSampleData>& testSamples,
    unsigned int seed) {

    std::cout << "\n-> Dividiendo dataset train/test (estratificado)" << std::endl;
    
    if (CONFIG_DATASET.usarDivisionManual) {
        std::cout << "   Division MANUAL: " << CONFIG_DATASET.muestrasTrainPorHablante
                  << " train + " << CONFIG_DATASET.muestrasTestPorHablante
                  << " test por hablante" << std::endl;
    } else {
        std::cout << "   Ratio: " << (CONFIG_DATASET.trainRatio * 100) << "% train / "
            << ((1 - CONFIG_DATASET.trainRatio) * 100) << "% test" << std::endl;
    }
    std::cout << "   Seed: " << seed << std::endl;

    std::mt19937 gen(seed);

    for (const auto& [hablante, archivos] : archivosPorHablante) {
        std::vector<fs::path> archivosShuffled = archivos;
        std::shuffle(archivosShuffled.begin(), archivosShuffled.end(), gen);

        size_t n_train;
        size_t n_test;
        
        if (CONFIG_DATASET.usarDivisionManual) {
            // Division manual: usar valores exactos de config
            n_train = std::min(static_cast<size_t>(CONFIG_DATASET.muestrasTrainPorHablante),
                              archivosShuffled.size());
            n_test = std::min(static_cast<size_t>(CONFIG_DATASET.muestrasTestPorHablante),
                             archivosShuffled.size() - n_train);
            
            size_t n_total_requerido = CONFIG_DATASET.muestrasTrainPorHablante + CONFIG_DATASET.muestrasTestPorHablante;
            if (archivosShuffled.size() < n_total_requerido) {
                std::cerr << "ADVERTENCIA: Hablante " << hablante << " tiene solo "
                         << archivosShuffled.size() << " audios, se requieren "
                         << n_total_requerido << std::endl;
            }
        } else {
            // Division por ratio
            n_train = static_cast<size_t>(archivosShuffled.size() * CONFIG_DATASET.trainRatio);
            if (n_train == 0) n_train = 1;
            if (n_train >= archivosShuffled.size()) n_train = archivosShuffled.size() - 1;
            n_test = archivosShuffled.size() - n_train;
        }

        std::cout << "   Hablante " << std::setw(5) << hablante << ": "
            << std::setw(3) << n_train << " train, "
            << std::setw(3) << n_test << " test" << std::endl;

        // Solo procesar la cantidad exacta especificada
        size_t n_total = n_train + n_test;
        for (size_t i = 0; i < n_total && i < archivosShuffled.size(); ++i) {
            if (i < n_train) {
                trainSamples.emplace_back(archivosShuffled[i], hablante);
            }
            else {
                testSamples.emplace_back(archivosShuffled[i], hablante);
            }
        }
    }

    std::cout << "   @ Split: " << trainSamples.size() << " train, "
        << testSamples.size() << " test" << std::endl;
}

// PROCESAMIENTO PARALELO USANDO PIPELINE

void procesarMuestrasParalelo(
    std::vector<AudioSampleData>& samples,
    const std::string& tipo,
    EstadisticasProcesamiento& stats,
    MetricasRendimiento* metricas = nullptr) {

    std::cout << "\n-> Procesando " << samples.size() << " archivos [" << tipo << "]"
        << std::endl;

    // Informar modo de procesamiento
    const bool usaAugmentation = CONFIG_DATASET.usarAugmentation &&
        (CONFIG_AUG.numVariaciones > 0);

    if (usaAugmentation) {
        std::cout << "   Modo: CON augmentation ("
            << (CONFIG_AUG.numVariaciones + 1) << " muestras por audio)"
            << std::endl;
    }
    else {
        std::cout << "   Modo: SIN augmentation (1 muestra por audio)" << std::endl;
    }

    // Profiling
    ProfilerEtapa profiler("Procesamiento_" + tipo);
    std::thread monitorThread;

    if (CONFIG_PROFILING.habilitado) {
        profiler.iniciar();
        monitorThread = std::thread(monitorearRecursos, &profiler);
    }

    int total = static_cast<int>(samples.size());
    std::atomic<int> procesados{ 0 };

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; ++i) {
        //  USAR PIPELINE: Una sola llamada procesa todo
        std::vector<std::vector<AudioSample>> allFeatures;
        bool exito = procesarAudioCompleto(samples[i].path, allFeatures);

        if (exito && !allFeatures.empty()) {
            stats.totalExitosos++;
            stats.audiosCrudos++;
            stats.muestrasGeneradas += static_cast<int>(allFeatures.size());

#pragma omp critical
            {
                stats.muestrasPorHablante[samples[i].etiqueta] +=
                    static_cast<int>(allFeatures.size());

                // Expandir samples con todas las muestras generadas
                for (size_t v = 0; v < allFeatures.size(); ++v) {
                    samples.emplace_back(
                        samples[i].path,
                        samples[i].etiqueta,
                        std::move(allFeatures[v])
                    );
                }
            }
        }
        else {
            stats.totalFallidos++;
        }

        stats.totalProcesados++;
        int actual = ++procesados;

        if (actual % 10 == 0 || actual == total) {
#pragma omp critical
            {
                stats.mostrarProgreso(actual, total,
                    samples[i].path.filename().string());
            }
        }
    }

    // Eliminar muestras originales (no procesadas)
    samples.erase(
        std::remove_if(samples.begin(), samples.begin() + total,
            [](const AudioSampleData& s) { return !s.procesado; }),
        samples.begin() + total
    );

    std::cout << "   & Completado: " << stats.audiosCrudos.load() << " audios → "
        << stats.muestrasGeneradas.load() << " muestras" << std::endl;

    // Finalizar profiling
    if (CONFIG_PROFILING.habilitado) {
        profiler.detener();
        if (monitorThread.joinable()) monitorThread.join();
        if (metricas) *metricas = profiler.obtenerMetricas();
    }
}

// CONVERSION Y GUARDADO

Dataset convertirADataset(const std::vector<AudioSampleData>& samples) {
    Dataset dataset;

    for (const auto& sample : samples) {
        if (sample.procesado && !sample.features.empty()) {
            dataset.X.push_back(sample.features);
            dataset.y.push_back(sample.etiqueta);
        }
    }

    return dataset;
}

// MAIN

int main(int argc, char* argv[]) {
    std::cout << std::string(70, '*') << std::endl;
    std::cout << "*  PROCESADOR DE DATASET - SISTEMA BIOMETRICO DE VOZ  *" << std::endl;
    std::cout << std::string(70, '*') << std::endl;

    // Configuración
   // std::string datasetPath = "D:\\muchosDataset\\Dataset\\audio7";
  // std::string datasetPath = "Dataset/audio7";
   std::string datasetPath = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V1\\mls_spanish\\train\\audio";
 //   std::string datasetPath = "D:\\8vo-Nivel\\Tesiss\\DatasetReal";
 // std::string datasetPath = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V1\\mls_spanish\\audio";
   //std::string datasetPath = "D:\\8vo-Nivel\\Tesiss\\DATASET\\DatasetAplicada\\V3\\train";
    std::string outputDir = "processed_dataset_bin";
    int seed = CONFIG_DATASET.seed;

    if (argc >= 2) datasetPath = argv[1];
    if (argc >= 3) outputDir = argv[2];
    if (argc >= 4) seed = std::stoi(argv[3]);

    std::cout << "\n-> Configuracion:" << std::endl;
    std::cout << "   Dataset: " << datasetPath << std::endl;
    std::cout << "   Salida: " << outputDir << std::endl;
    std::cout << "   Features: " << CONFIG_MFCC.totalFeatures << std::endl;
    std::cout << "   Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "   Augmentation: "
        << (CONFIG_DATASET.usarAugmentation ? "SI" : "NO") << std::endl;

    if (CONFIG_DATASET.usarAugmentation) {
        std::cout << "   Variaciones: " << (CONFIG_AUG.numVariaciones + 1) << std::endl;
    }

    fs::create_directories(outputDir);

    // 1. Recopilar archivos con mapeo de nombres
    std::map<int, std::string> idANombre;
    auto archivosPorHablante = recopilarArchivosPorHablante(datasetPath, idANombre);
    if (archivosPorHablante.empty()) {
        std::cerr << "! Error: No hay archivos validos" << std::endl;
        return -1;
    }

    // Detectar tipo de dataset
    bool datasetConNombres = false;
    for (const auto& [id, nombre] : idANombre) {
        try {
            std::stoi(nombre);
        } catch (...) {
            datasetConNombres = true;
            break;
        }
    }

    if (datasetConNombres) {
        std::cout << "\n-> Tipo de dataset: NOMBRES (IDs asignados automaticamente)" << std::endl;
    } else {
        std::cout << "\n-> Tipo de dataset: IDS NUMERICOS" << std::endl;
    }

    // Mostrar distribución
    std::cout << "\n-> Distribucion inicial:" << std::endl;
    int totalArchivos = 0;
    for (const auto& [h, archivos] : archivosPorHablante) {
        std::string nombre = idANombre[h];
        std::cout << "   Hablante " << std::setw(5) << h << " (" << std::setw(20) << nombre << "): "
            << std::setw(2) << archivos.size() << " archivos" << std::endl;
        totalArchivos += static_cast<int>(archivos.size());
    }
    std::cout << "   @ Total: " << totalArchivos << " archivos, "
        << archivosPorHablante.size() << " hablantes" << std::endl;

    // 2. Dividir train/test
    std::vector<AudioSampleData> trainSamples, testSamples;
    dividirTrainTest(archivosPorHablante, trainSamples, testSamples, seed);

    // 3. Procesar usando PIPELINE (maneja augmentation automáticamente)
    EstadisticasProcesamiento statsTrain, statsTest;
    MetricasRendimiento metricasTrain, metricasTest;

    procesarMuestrasParalelo(trainSamples, "TRAIN", statsTrain, &metricasTrain);
    procesarMuestrasParalelo(testSamples, "TEST", statsTest, &metricasTest);

    if (trainSamples.empty() || testSamples.empty()) {
        std::cerr << "! Error: Insuficientes muestras procesadas" << std::endl;
        return -1;
    }

    // 4. Convertir y guardar
    std::cout << "\n-> Guardando datasets..." << std::endl;

    Dataset datasetTrain = convertirADataset(trainSamples);
    Dataset datasetTest = convertirADataset(testSamples);

    std::string trainPath = obtenerRutaDatasetTrain();
    std::string testPath = obtenerRutaDatasetTest();

    if (!guardarDatasetBinario(trainPath, datasetTrain)) {
        std::cerr << "! Error guardando train" << std::endl;
        return -1;
    }

    if (!guardarDatasetBinario(testPath, datasetTest)) {
        std::cerr << "! Error guardando test" << std::endl;
        return -1;
    }

    std::cout << "   & Train: " << trainPath << std::endl;
    std::cout << "   & Test: " << testPath << std::endl;

    // 5. Mapeo ahora se genera automaticamente en metadata.json durante el entrenamiento
    std::cout << "   @ Mapeo se generara en metadata.json durante entrenamiento (" 
              << idANombre.size() << " hablantes)" << std::endl;

    // Resumen final
    std::cout << "\n" << std::string(70, '*') << std::endl;
    std::cout << "*  PROCESAMIENTO COMPLETADO  *" << std::endl;
    std::cout << std::string(70, '*') << std::endl;

    std::cout << "\n@ RESUMEN:" << std::endl;
    std::cout << "   Archivos train: " << statsTrain.audiosCrudos.load() << " → "
        << trainSamples.size() << " muestras" << std::endl;
    std::cout << "   Archivos test: " << statsTest.audiosCrudos.load() << " → "
        << testSamples.size() << " muestras" << std::endl;
    std::cout << "   Hablantes: " << idANombre.size() << std::endl;

    if (CONFIG_DATASET.usarAugmentation) {
        float factor = (float)statsTrain.muestrasGeneradas / statsTrain.audiosCrudos.load();
        std::cout << "   Factor aumento train: x" << std::fixed << std::setprecision(2)
            << factor << std::endl;
    }

    // RESUMEN DE PROFILING
    if (CONFIG_PROFILING.habilitado) {
        std::cout << "\n" << std::string(70, '*') << std::endl;
        std::cout << "*  RESUMEN DE PROFILING DE RENDIMIENTO  *" << std::endl;
        std::cout << std::string(70, '*') << std::endl;

        std::cout << "\n# PROCESAMIENTO TRAIN:" << std::endl;
        if (CONFIG_PROFILING.medirTiempo) {
            std::cout << "   Tiempo: " << std::fixed << std::setprecision(2)
                      << (metricasTrain.tiempoMs / 1000.0) << " segundos"
                      << " (" << (metricasTrain.tiempoMs / statsTrain.audiosCrudos.load())
                      << " ms/audio)" << std::endl;
        }
        if (CONFIG_PROFILING.medirRAM) {
            std::cout << "   RAM Peak: " << std::fixed << std::setprecision(1)
                      << metricasTrain.ramPeakMB << " MB" << std::endl;
            std::cout << "   RAM Promedio: " << std::fixed << std::setprecision(1)
                      << metricasTrain.ramPromMB << " MB" << std::endl;
        }
        if (CONFIG_PROFILING.medirCPU) {
            std::cout << "   CPU Promedio: " << std::fixed << std::setprecision(1)
                      << metricasTrain.cpuProm << " %" << std::endl;
        }

        std::cout << "\n# PROCESAMIENTO TEST:" << std::endl;
        if (CONFIG_PROFILING.medirTiempo) {
            std::cout << "   Tiempo: " << std::fixed << std::setprecision(2)
                      << (metricasTest.tiempoMs / 1000.0) << " segundos"
                      << " (" << (metricasTest.tiempoMs / statsTest.audiosCrudos.load())
                      << " ms/audio)" << std::endl;
        }
        if (CONFIG_PROFILING.medirRAM) {
            std::cout << "   RAM Peak: " << std::fixed << std::setprecision(1)
                      << metricasTest.ramPeakMB << " MB" << std::endl;
            std::cout << "   RAM Promedio: " << std::fixed << std::setprecision(1)
                      << metricasTest.ramPromMB << " MB" << std::endl;
        }
        if (CONFIG_PROFILING.medirCPU) {
            std::cout << "   CPU Promedio: " << std::fixed << std::setprecision(1)
                      << metricasTest.cpuProm << " %" << std::endl;
        }

        double tiempoTotalS = (metricasTrain.tiempoMs + metricasTest.tiempoMs) / 1000.0;
        double ramMaxTotal = std::max(metricasTrain.ramPeakMB, metricasTest.ramPeakMB);

        std::cout << "\n# TOTALES:" << std::endl;
        std::cout << "   Tiempo procesamiento: " << std::fixed << std::setprecision(2)
                  << tiempoTotalS << " segundos" << std::endl;
        std::cout << "   RAM Peak global: " << std::fixed << std::setprecision(1)
                  << ramMaxTotal << " MB" << std::endl;
        std::cout << "   Throughput: " << std::fixed << std::setprecision(2)
                  << ((statsTrain.audiosCrudos.load() + statsTest.audiosCrudos.load()) / tiempoTotalS)
                  << " audios/segundo" << std::endl;
    }

    return 0;
}