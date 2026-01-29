#ifndef PCA_UTILS_H
#define PCA_UTILS_H

#include <vector>
#include <string>

struct ModeloPCA {
    std::vector<std::vector<double>> componentes;
    std::vector<double> medias;
};

// Entrena modelo PCA completo (media + componentes)
ModeloPCA entrenarPCA(const std::vector<std::vector<double>>& datos, int numComponentes);

// Proyecta datos usando un modelo PCA existente
std::vector<std::vector<double>> aplicarPCAConModelo(
    const std::vector<std::vector<double>>& datos,
    const ModeloPCA& modelo
);

// Aplica PCA directamente desde archivo de modelo
bool aplicarPCADesdeModelo(
    const std::string& rutaModelo,
    const std::vector<std::vector<double>>& datosEntrada,
    std::vector<std::vector<double>>& salidaPCA
);

// Guarda/carga modelo PCA en CSV plano (media + componentes)
bool guardarModeloPCA(const std::string& ruta, const ModeloPCA& modelo);
ModeloPCA cargarModeloPCA(const std::string& ruta);

#endif 
