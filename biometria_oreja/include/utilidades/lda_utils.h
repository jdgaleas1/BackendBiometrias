#ifndef LDA_UTILS_H
#define LDA_UTILS_H

#include <vector>
#include <string>

struct ModeloLDA {
    std::vector<std::vector<double>> componentes;  // Eigenvectors (proyecciones)
    std::vector<double> mediaGlobal;               // Media global de entrenamiento
    int numClases;                                  // Numero de clases
};

// Entrena modelo LDA (Fisher Linear Discriminant)
// Entrada: datos ya reducidos por PCA, etiquetas de clase
// numComponentes: max = numClases - 1 (por defecto usa el maximo)
ModeloLDA entrenarLDA(
    const std::vector<std::vector<double>>& datos,
    const std::vector<int>& etiquetas,
    int numComponentes = -1  // -1 = usar maximo (numClases - 1)
);

// Proyecta datos usando modelo LDA existente
std::vector<std::vector<double>> aplicarLDAConModelo(
    const std::vector<std::vector<double>>& datos,
    const ModeloLDA& modelo
);

// Guarda/carga modelo LDA
bool guardarModeloLDA(const std::string& ruta, const ModeloLDA& modelo);
ModeloLDA cargarModeloLDA(const std::string& ruta);

#endif
