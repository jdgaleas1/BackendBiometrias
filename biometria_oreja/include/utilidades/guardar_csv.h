#ifndef GUARDAR_CSV_H
#define GUARDAR_CSV_H

#include <string>
#include <vector>

// Versión principal: permite elegir delimitador y retorna éxito/fracaso
bool guardarCSV(const std::string& rutaArchivo,
                const std::vector<std::vector<double>>& datos,
                const std::vector<int>& etiquetas,
                char delimitador);

// Sobrecarga compatible: si no pasas delimitador, usa ';'
inline bool guardarCSV(const std::string& rutaArchivo,
                       const std::vector<std::vector<double>>& datos,
                       const std::vector<int>& etiquetas) {
    return guardarCSV(rutaArchivo, datos, etiquetas, ';');
}

#endif
