#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <string>
#include <vector>
#include <map>

// Carga rutas y genera etiquetas internas consecutivas.
void cargarRutasDataset(const std::string& carpetaBase, std::vector<std::string>& rutas, std::vector<int>& etiquetas, std::map<int, int>& mapaEtiquetaRealAInterna);

#endif
