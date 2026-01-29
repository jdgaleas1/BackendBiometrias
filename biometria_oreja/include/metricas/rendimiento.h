#ifndef RENDIMIENTO_H
#define RENDIMIENTO_H

#include <string>
#include <unordered_map>
#include <cstddef>

class MedidorRendimiento {
public:
    explicit MedidorRendimiento(const std::string& nombreProceso);

    void iniciar();
    void marcar(const std::string& fase);
    void finalizar();

    void imprimirResumen() const;

    void guardarEnArchivo(const std::string& rutaCSV) const;
    void guardarFasesCSV(const std::string& rutaCSV) const;

private:
    std::string nombre;

    double tiempoSegundos;
    double cpuSegundos;
    double cpuPorcEquivalente;

    size_t memoriaKB;
    size_t picoMemoriaKB;

    long long t0_ns;
    long long tFase_ns;

    std::string faseActual;

    std::unordered_map<std::string, double> fasesSeg;
    std::unordered_map<std::string, double> fasesCpuSeg;
    std::unordered_map<std::string, size_t> fasesRamMaxKB;

    double cpuLastSeg;
    size_t memLastKB;
    size_t peakLastKB;
};

#endif
