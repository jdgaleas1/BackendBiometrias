#ifndef CARGAR_CSV_H
#define CARGAR_CSV_H

#include <string>
#include <vector>

bool cargarCSV(const std::string& ruta,
    std::vector<std::vector<double>>& caracteristicas,
    std::vector<int>& etiquetas,
    char delimitador = ';');

bool cargarCSVSinEtiquetas(const std::string& ruta,
    std::vector<std::vector<double>>& caracteristicas,
    char delimitador = ';');

#endif
