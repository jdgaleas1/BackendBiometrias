#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <set>

// ============================================================================
// VALIDACION DE DATASETS
// ============================================================================

bool validarDataset(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y) {

    // Verificar que no esten vacíos
    if (X.empty() || y.empty()) {
        std::cerr << "! Error: Dataset vacio" << std::endl;
        return false;
    }

    // Verificar mismo tamaño
    if (X.size() != y.size()) {
        std::cerr << "! Error: X e y tienen tamanos diferentes ("
            << X.size() << " vs " << y.size() << ")" << std::endl;
        return false;
    }

    // Verificar dimensión consistente
    size_t dim = X[0].size();
    if (dim == 0) {
        std::cerr << "! Error: Dimension de caracteristicas es 0" << std::endl;
        return false;
    }

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != dim) {
            std::cerr << "! Error: Dimension inconsistente en muestra " << i
                << " (esperado " << dim << ", encontrado " << X[i].size() << ")"
                << std::endl;
            return false;
        }
    }

    // Verificar valores válidos (no NaN, no Inf)
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (std::isnan(X[i][j]) || std::isinf(X[i][j])) {
                std::cerr << "! Error: Valor invalido (NaN/Inf) en muestra " << i
                    << ", feature " << j << std::endl;
                return false;
            }
        }
    }

    // Verificar que haya al menos 1 muestra por clase
    std::map<int, int> counts;
    for (int label : y) {
        counts[label]++;
    }

    for (const auto& [clase, count] : counts) {
        if (count == 0) {
            std::cerr << "! Error: Clase " << clase << " sin muestras" << std::endl;
            return false;
        }
    }

    return true;
}

bool validarDataset(const Dataset& dataset) {
    return validarDataset(dataset.X, dataset.y);
}

// ============================================================================
// ESTADISTICAS DE DATASETS
// ============================================================================

DatasetStats calcularEstadisticas(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y) {

    DatasetStats stats;
    stats.total_muestras = X.size();
    stats.dimension = X.empty() ? 0 : X[0].size();
    stats.tiene_valores_invalidos = false;

    if (X.empty() || y.empty()) {
        stats.num_clases = 0;
        stats.ratio_desbalance = 0.0;
        stats.rango_min_features = 0.0;
        stats.rango_max_features = 0.0;
        return stats;
    }

    // Contar muestras por clase
    for (int label : y) {
        stats.muestras_por_clase[label]++;
    }
    stats.num_clases = stats.muestras_por_clase.size();

    // Calcular ratio de desbalance
    int min_muestras = std::numeric_limits<int>::max();
    int max_muestras = 0;

    for (const auto& [clase, count] : stats.muestras_por_clase) {
        min_muestras = std::min(min_muestras, count);
        max_muestras = std::max(max_muestras, count);
    }

    stats.ratio_desbalance = (min_muestras > 0)
        ? static_cast<AudioSample>(max_muestras) / min_muestras : 0.0;

    // Verificar valores inválidos
    for (const auto& sample : X) {
        for (AudioSample val : sample) {
            if (std::isnan(val) || std::isinf(val)) {
                stats.tiene_valores_invalidos = true;
                break;
            }
        }
        if (stats.tiene_valores_invalidos) break;
    }

    // Calcular rangos de features
    size_t dim = stats.dimension;
    std::vector<AudioSample> mins(dim, std::numeric_limits<double>::max());
    std::vector<AudioSample> maxs(dim, std::numeric_limits<double>::lowest());

    for (const auto& sample : X) {
        for (size_t j = 0; j < dim; ++j) {
            mins[j] = std::min(mins[j], sample[j]);
            maxs[j] = std::max(maxs[j], sample[j]);
        }
    }

    stats.rango_min_features = std::numeric_limits<double>::max();
    stats.rango_max_features = 0.0;

    for (size_t j = 0; j < dim; ++j) {
        AudioSample rango = maxs[j] - mins[j];
        stats.rango_min_features = std::min(stats.rango_min_features, rango);
        stats.rango_max_features = std::max(stats.rango_max_features, rango);
    }

    return stats;
}

DatasetStats calcularEstadisticas(const Dataset& dataset) {
    return calcularEstadisticas(dataset.X, dataset.y);
}

void mostrarEstadisticas(const DatasetStats& stats, const std::string& nombre) {
    std::cout << "\n-> Estadisticas del dataset: " << nombre << std::endl;
    std::cout << "   " << std::string(60, '=') << std::endl;

    std::cout << "\n   Informacion basica:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Total muestras:      " << stats.total_muestras << std::endl;
    std::cout << "   Dimension:           " << stats.dimension << " features" << std::endl;
    std::cout << "   Clases:              " << stats.num_clases << std::endl;

    std::cout << "\n   Distribucion de clases:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;

    for (const auto& [clase, count] : stats.muestras_por_clase) {
        AudioSample porcentaje = 100.0 * count / stats.total_muestras;
        std::cout << "   Clase " << std::setw(5) << clase << ": "
            << std::setw(4) << count << " muestras ("
            << std::fixed << std::setprecision(1) << porcentaje << "%)"
            << std::endl;
    }

    std::cout << "\n   Desbalance de clases:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Ratio (max/min):     1:" << std::fixed << std::setprecision(2)
        << stats.ratio_desbalance << std::endl;

    if (stats.ratio_desbalance > 5.0) {
        std::cerr << "   % Warning: Dataset muy desbalanceado" << std::endl;
    }

    std::cout << "\n   Rangos de features:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;
    std::cout << "   Rango minimo:        " << std::fixed << std::setprecision(4)
        << stats.rango_min_features << std::endl;
    std::cout << "   Rango maximo:        " << stats.rango_max_features << std::endl;

    std::cout << "\n   Validacion:" << std::endl;
    std::cout << "   " << std::string(60, '-') << std::endl;

    if (stats.tiene_valores_invalidos) {
        std::cerr << "   ! ERROR: Dataset contiene NaN o Inf" << std::endl;
    }
    else {
        std::cout << "   @ OK: Todos los valores son validos" << std::endl;
    }

    std::cout << "   " << std::string(60, '=') << std::endl;
}

// ============================================================================
// MAPEO DE SPEAKERS
// ============================================================================

SpeakerMapping crearMapeoSpeakers(const std::vector<int>& y) {
    SpeakerMapping mapping;

    if (y.empty()) {
        std::cerr << "! Error: Vector de etiquetas vacio" << std::endl;
        return mapping;
    }

    std::cout << "-> Creando mapeo de speakers" << std::endl;

    // Obtener IDs únicos ordenados
    std::set<int> unique_ids(y.begin(), y.end());
    std::vector<int> sorted_ids(unique_ids.begin(), unique_ids.end());
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Crear mapeo bidireccional
    int index = 0;
    for (int speaker_id : sorted_ids) {
        mapping.speaker_to_index[speaker_id] = index;
        mapping.index_to_speaker[index] = speaker_id;
        index++;
    }

    std::cout << "   & Mapeo creado: " << mapping.size() << " speakers" << std::endl;
    std::cout << "   IDs originales: ";

    int count = 0;
    for (int id : sorted_ids) {
        if (count++ > 0) std::cout << ", ";
        std::cout << id;
        if (count >= 10) {
            std::cout << "...";
            break;
        }
    }
    std::cout << std::endl;

    return mapping;
}

void aplicarMapeoSpeakers(std::vector<int>& y, const SpeakerMapping& mapping) {
    if (y.empty()) return;

    std::cout << "-> Aplicando mapeo de speakers" << std::endl;

    int errores = 0;

    for (int& label : y) {
        auto it = mapping.speaker_to_index.find(label);
        if (it != mapping.speaker_to_index.end()) {
            label = it->second;
        }
        else {
            std::cerr << "% Warning: Speaker ID " << label << " no encontrado en mapeo"
                << std::endl;
            errores++;
        }
    }

    if (errores > 0) {
        std::cerr << "   ! " << errores << " etiquetas no mapeadas" << std::endl;
    }
    else {
        std::cout << "   & Mapeo aplicado: " << y.size() << " etiquetas convertidas"
            << std::endl;
    }
}

// ============================================================================
// COMPATIBILIDAD Y FUSION
// ============================================================================

bool verificarCompatibilidad(const Dataset& train, const Dataset& test) {
    if (train.empty() || test.empty()) {
        std::cerr << "! Error: Alguno de los datasets esta vacio" << std::endl;
        return false;
    }

    // Verificar misma dimensión
    if (train.dim() != test.dim()) {
        std::cerr << "! Error: Dimensiones diferentes (train: " << train.dim()
            << ", test: " << test.dim() << ")" << std::endl;
        return false;
    }

    // Verificar que test no tenga clases no vistas en train
    std::set<int> clases_train(train.y.begin(), train.y.end());
    std::set<int> clases_test(test.y.begin(), test.y.end());

    std::vector<int> clases_no_vistas;
    for (int clase : clases_test) {
        if (clases_train.find(clase) == clases_train.end()) {
            clases_no_vistas.push_back(clase);
        }
    }

    if (!clases_no_vistas.empty()) {
        std::cerr << "! Error: Test tiene clases no vistas en train: ";
        for (size_t i = 0; i < clases_no_vistas.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << clases_no_vistas[i];
        }
        std::cerr << std::endl;
        return false;
    }

    std::cout << "@ Datasets compatibles (dim=" << train.dim()
        << ", clases=" << clases_train.size() << ")" << std::endl;

    return true;
}

Dataset fusionarDatasets(const std::vector<Dataset>& datasets) {
    Dataset fusionado;

    if (datasets.empty()) {
        std::cerr << "! Error: Lista de datasets vacia" << std::endl;
        return fusionado;
    }

    std::cout << "-> Fusionando " << datasets.size() << " datasets" << std::endl;

    // Validar que todos tengan la misma dimensión
    size_t dim = datasets[0].dim();
    for (size_t i = 1; i < datasets.size(); ++i) {
        if (datasets[i].dim() != dim) {
            std::cerr << "! Error: Dataset " << i << " tiene dimension diferente ("
                << datasets[i].dim() << " vs " << dim << ")" << std::endl;
            return Dataset();
        }
    }

    // Fusionar
    for (const auto& dataset : datasets) {
        fusionado.X.insert(fusionado.X.end(), dataset.X.begin(), dataset.X.end());
        fusionado.y.insert(fusionado.y.end(), dataset.y.begin(), dataset.y.end());
    }

    std::cout << "   & Fusion completada: " << fusionado.size() << " muestras totales"
        << std::endl;

    return fusionado;
}

// ============================================================================
// FILTRADO Y SUBSET
// ============================================================================

Dataset filtrarPorClases(const Dataset& dataset,
    const std::vector<int>& clases_incluir) {

    Dataset filtrado;

    if (dataset.empty()) {
        std::cerr << "! Error: Dataset vacio" << std::endl;
        return filtrado;
    }

    std::cout << "-> Filtrando dataset por " << clases_incluir.size() << " clases"
        << std::endl;

    std::set<int> clases_set(clases_incluir.begin(), clases_incluir.end());

    for (size_t i = 0; i < dataset.size(); ++i) {
        if (clases_set.find(dataset.y[i]) != clases_set.end()) {
            filtrado.X.push_back(dataset.X[i]);
            filtrado.y.push_back(dataset.y[i]);
        }
    }

    std::cout << "   & Filtrado completado: " << filtrado.size() << " muestras "
        << "(de " << dataset.size() << " originales)" << std::endl;

    return filtrado;
}

Dataset crearSubset(const Dataset& dataset, size_t n_samples, unsigned int seed) {
    Dataset subset;

    if (dataset.empty()) {
        std::cerr << "! Error: Dataset vacio" << std::endl;
        return subset;
    }

    if (n_samples >= dataset.size()) {
        std::cerr << "% Warning: n_samples >= tamano del dataset, retornando dataset completo"
            << std::endl;
        return dataset;
    }

    std::cout << "-> Creando subset de " << n_samples << " muestras (seed=" << seed << ")"
        << std::endl;

    // Crear indices y shuffle
    std::vector<size_t> indices(dataset.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Tomar primeras n_samples
    for (size_t i = 0; i < n_samples; ++i) {
        size_t idx = indices[i];
        subset.X.push_back(dataset.X[idx]);
        subset.y.push_back(dataset.y[idx]);
    }

    std::cout << "   & Subset creado: " << subset.size() << " muestras" << std::endl;

    return subset;
}
