#include "metricas/rendimiento.h"

#define NOMINMAX

#include <chrono>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#endif

using namespace std;
using namespace std::chrono;

static void capturarSnapshot(double& cpuSeg, size_t& memKB, size_t& peakKB) {
#if defined(_WIN32)
    cpuSeg = 0.0;
    memKB = 0;
    peakKB = 0;

    HANDLE h = GetCurrentProcess();

    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(h, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        memKB = (size_t)(pmc.WorkingSetSize / 1024);
        peakKB = (size_t)(pmc.PeakWorkingSetSize / 1024);
    }

    FILETIME ftCreate{}, ftExit{}, ftKernel{}, ftUser{};
    if (GetProcessTimes(h, &ftCreate, &ftExit, &ftKernel, &ftUser)) {
        ULARGE_INTEGER k{}, u{};
        k.LowPart = ftKernel.dwLowDateTime; k.HighPart = ftKernel.dwHighDateTime;
        u.LowPart = ftUser.dwLowDateTime;   u.HighPart = ftUser.dwHighDateTime;
        cpuSeg = (k.QuadPart + u.QuadPart) * 1e-7;
    }
#else
    cpuSeg = 0.0;
    memKB = 0;
    peakKB = 0;

    struct rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        cpuSeg = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec
            + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
        peakKB = (size_t)ru.ru_maxrss;
    }

    FILE* f = std::fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "VmRSS:", 6) == 0) {
                long kb = 0;
                if (std::sscanf(line + 6, "%ld", &kb) == 1) memKB = (size_t)kb;
                break;
            }
        }
        std::fclose(f);
    }
#endif
}

MedidorRendimiento::MedidorRendimiento(const std::string& nombreProceso)
    : nombre(nombreProceso),
      tiempoSegundos(0.0),
      cpuSegundos(0.0),
      cpuPorcEquivalente(0.0),
      memoriaKB(0),
      picoMemoriaKB(0),
      t0_ns(0),
      tFase_ns(0),
      faseActual(),
      cpuLastSeg(0.0),
      memLastKB(0),
      peakLastKB(0) {
}

void MedidorRendimiento::iniciar() {
    t0_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    tFase_ns = 0;
    faseActual.clear();

    fasesSeg.clear();
    fasesCpuSeg.clear();
    fasesRamMaxKB.clear();

    capturarSnapshot(cpuLastSeg, memLastKB, peakLastKB);
}

void MedidorRendimiento::marcar(const std::string& fase) {
    const auto ahora_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

    double cpuNow = 0.0;
    size_t memNow = 0, peakNow = 0;
    capturarSnapshot(cpuNow, memNow, peakNow);

    if (!faseActual.empty() && tFase_ns > 0) {
        double dt = (ahora_ns - tFase_ns) / 1e9;
        fasesSeg[faseActual] += dt;

        double dCpu = cpuNow - cpuLastSeg;
        if (dCpu < 0) dCpu = 0.0;
        fasesCpuSeg[faseActual] += dCpu;

        size_t cand = std::max(memNow, peakNow);
        fasesRamMaxKB[faseActual] = std::max(fasesRamMaxKB[faseActual], cand);
    }

    faseActual = fase;
    tFase_ns = ahora_ns;

    cpuLastSeg = cpuNow;
    memLastKB = memNow;
    peakLastKB = peakNow;
}

void MedidorRendimiento::finalizar() {
    const auto fin_ns = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

    double cpuNow = 0.0;
    size_t memNow = 0, peakNow = 0;
    capturarSnapshot(cpuNow, memNow, peakNow);

    if (!faseActual.empty() && tFase_ns > 0) {
        double dt = (fin_ns - tFase_ns) / 1e9;
        fasesSeg[faseActual] += dt;

        double dCpu = cpuNow - cpuLastSeg;
        if (dCpu < 0) dCpu = 0.0;
        fasesCpuSeg[faseActual] += dCpu;

        size_t cand = std::max(memNow, peakNow);
        fasesRamMaxKB[faseActual] = std::max(fasesRamMaxKB[faseActual], cand);
    }

    tiempoSegundos = (fin_ns - t0_ns) / 1e9;

    cpuSegundos = cpuNow;
    memoriaKB = memNow;
    picoMemoriaKB = peakNow;

    cpuPorcEquivalente = (tiempoSegundos > 0.0) ? (100.0 * cpuSegundos / tiempoSegundos) : 0.0;
}

void MedidorRendimiento::imprimirResumen() const {
    cout << "\n[RENDIMIENTO] " << nombre << "\n"
         << "Tiempo total (s): " << tiempoSegundos << "\n"
         << "CPU total (s):   " << cpuSegundos << "\n"
         << "CPU eq. (%):     " << cpuPorcEquivalente << "\n"
         << "RAM actual (KB): " << memoriaKB << "\n"
         << "RAM pico  (KB):  " << picoMemoriaKB << "\n";

    for (const auto& kv : fasesSeg) {
        const string& fase = kv.first;
        double t = kv.second;

        double c = 0.0;
        auto itC = fasesCpuSeg.find(fase);
        if (itC != fasesCpuSeg.end()) c = itC->second;

        double cpuPct = (t > 0.0) ? (100.0 * c / t) : 0.0;

        size_t rm = 0;
        auto itM = fasesRamMaxKB.find(fase);
        if (itM != fasesRamMaxKB.end()) rm = itM->second;

        cout << "  - " << fase
             << ": " << t << " s"
             << " | CPU: " << c << " s"
             << " (" << cpuPct << "%)"
             << " | RAM_max(KB): " << rm
             << "\n";
    }
}

void MedidorRendimiento::guardarEnArchivo(const std::string& rutaCSV) const {
    ofstream f(rutaCSV, ios::app);
    if (!f) return;

    f << nombre << "," << tiempoSegundos << "," << cpuSegundos << ","
      << cpuPorcEquivalente << "," << memoriaKB << "," << picoMemoriaKB << "\n";
}

void MedidorRendimiento::guardarFasesCSV(const std::string& rutaCSV) const {
    ofstream f(rutaCSV, ios::app);
    if (!f) return;

    for (const auto& kv : fasesSeg) {
        const string& fase = kv.first;
        double t = kv.second;

        double c = 0.0;
        auto itC = fasesCpuSeg.find(fase);
        if (itC != fasesCpuSeg.end()) c = itC->second;

        double cpuPct = (t > 0.0) ? (100.0 * c / t) : 0.0;

        size_t rm = 0;
        auto itM = fasesRamMaxKB.find(fase);
        if (itM != fasesRamMaxKB.end()) rm = itM->second;

        f << nombre << "," << fase << "," << t << "," << c << "," << cpuPct << "," << rm << "\n";
    }
}
