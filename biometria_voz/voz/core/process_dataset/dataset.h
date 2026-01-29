#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <map>
#include "config.h"  // Para AudioSample

// ============================================================================
// ESTRUCTURAS PUBLICAS - VERSION 3.0 REFACTORIZADA
// Migracion: float -> AudioSample (double) para precision biometrica
// ============================================================================

/**
 * Estructura que representa un dataset de caracteristicas
 * Contiene matriz de features y vector de labels
 */
struct Dataset {
    std::vector<std::vector<AudioSample>> X;  // Matriz de caracteristicas [muestras][dim]
    std::vector<int> y;                        // Vector de etiquetas

    // Constructor vacio
    Dataset() = default;

    // Constructor con datos
    Dataset(const std::vector<std::vector<AudioSample>>& X_, const std::vector<int>& y_)
        : X(X_), y(y_) {
    }

    // Obtener numero de muestras
    size_t size() const { return X.size(); }

    // Obtener dimension de caracteristicas
    size_t dim() const { return X.empty() ? 0 : X[0].size(); }

    // Verificar si esta vacio
    bool empty() const { return X.empty() || y.empty(); }
};

/**
 * Resultado del split train/test
 */
struct SplitResult {
    Dataset train;
    Dataset test;

    // Estadisticas del split
    std::map<int, int> train_counts;  // Muestras por clase en train
    std::map<int, int> test_counts;   // Muestras por clase en test
};

/**
 * Mapeo de speaker IDs a indices secuenciales
 */
struct SpeakerMapping {
    std::map<int, int> speaker_to_index;  // Original ID -> Indice secuencial
    std::map<int, int> index_to_speaker;  // Indice secuencial -> Original ID

    // Numero de speakers
    size_t size() const { return speaker_to_index.size(); }

    // Verificar si existe un speaker
    bool contains(int speaker_id) const {
        return speaker_to_index.find(speaker_id) != speaker_to_index.end();
    }
};

/**
 * Estadisticas de un dataset
 */
struct DatasetStats {
    size_t total_muestras;
    size_t dimension;
    size_t num_clases;
    std::map<int, int> muestras_por_clase;
    AudioSample ratio_desbalance;         // max/min muestras
    bool tiene_valores_invalidos;         // NaN o Inf
    AudioSample rango_min_features;       // Rango minimo entre features
    AudioSample rango_max_features;       // Rango maximo entre features
};

// ============================================================================
// API I/O - CARGA Y GUARDADO DE DATASETS
// ============================================================================

/**
 * Carga un dataset desde archivo binario
 *
 * @param ruta Ruta del archivo binario del dataset
 * @param X Matriz de caracteristicas (salida)
 * @param y Vector de etiquetas (salida)
 * @return true si se cargo correctamente
 *
 * Formato binario por muestra: [dim:int][features:AudioSample*dim][label:int]
 */
bool cargarDatasetBinario(const std::string& ruta,
    std::vector<std::vector<AudioSample>>& X,
    std::vector<int>& y);

/**
 * Carga un dataset desde archivo binario (version con struct)
 *
 * @param ruta Ruta del archivo binario
 * @return Dataset cargado (vacio si error)
 */
Dataset cargarDatasetBinario(const std::string& ruta);

/**
 * Guarda un dataset en formato binario
 *
 * @param ruta Ruta del archivo de salida
 * @param X Matriz de caracteristicas
 * @param y Vector de etiquetas
 * @return true si se guardo correctamente
 */
bool guardarDatasetBinario(const std::string& ruta,
    const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y);

/**
 * Guarda un dataset en formato binario (version con struct)
 *
 * @param ruta Ruta del archivo de salida
 * @param dataset Dataset a guardar
 * @return true si se guardo correctamente
 */
bool guardarDatasetBinario(const std::string& ruta, const Dataset& dataset);

/**
 * Agrega nuevas muestras a un dataset binario existente (INCREMENTAL)
 * Solo escribe al final del archivo, sin recargar todo
 *
 * @param ruta Ruta del archivo binario existente
 * @param nuevas_X Matriz de caracteristicas nuevas
 * @param nuevas_y Vector de etiquetas nuevas
 * @return true si se agrego correctamente
 *
 * VENTAJAS:
 * - No recarga dataset completo en memoria
 * - Operacion O(N_nuevas) en vez de O(N_total)
 * - Escribe solo al final (append mode)
 *
 * Ejemplo:
 *   // Agregar 30 muestras del usuario 1005
 *   agregarMuestrasDataset("train.dat", nuevas_features, {1005, 1005, ...});
 */
bool agregarMuestrasDataset(const std::string& ruta,
    const std::vector<std::vector<AudioSample>>& nuevas_X,
    const std::vector<int>& nuevas_y);

// ============================================================================
// API SPLIT - DIVISION TRAIN/TEST
// ============================================================================

/**
 * Divide dataset en train/test de forma estratificada
 * Mantiene proporcion de clases en ambos conjuntos
 *
 * @param X Matriz de caracteristicas completa
 * @param y Vector de etiquetas completo
 * @param train_ratio Porcentaje para entrenamiento (0.0-1.0)
 * @param seed Semilla para reproducibilidad
 * @return SplitResult con train y test
 *
 * Estrategia:
 * - Split estratificado: mantiene proporcion de clases
 * - Shuffle aleatorio dentro de cada clase
 * - Valida que cada clase tenga muestras en ambos conjuntos
 *
 * Ejemplo:
 *   auto split = dividirTrainTest(X, y, 0.8, 42);
 *   entrenarModelo(split.train.X, split.train.y);
 *   evaluarModelo(split.test.X, split.test.y);
 */
SplitResult dividirTrainTest(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y,
    AudioSample train_ratio,
    unsigned int seed);

/**
 * Divide dataset en train/test (version con struct Dataset)
 *
 * @param dataset Dataset completo
 * @param train_ratio Porcentaje para entrenamiento
 * @param seed Semilla para reproducibilidad
 * @return SplitResult con train y test
 */
SplitResult dividirTrainTest(const Dataset& dataset,
    AudioSample train_ratio,
    unsigned int seed);

// ============================================================================
// API UTILS - VALIDACION Y UTILIDADES
// ============================================================================

/**
 * Valida la integridad de un dataset
 *
 * @param X Matriz de caracteristicas
 * @param y Vector de etiquetas
 * @return true si el dataset es valido
 *
 * Validaciones:
 * - X e y tienen el mismo tamaño
 * - Todas las muestras tienen la misma dimension
 * - No hay valores NaN o Inf
 * - Hay al menos 1 muestra por clase
 * - Dimension > 0
 */
bool validarDataset(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y);

/**
 * Valida dataset (version con struct)
 *
 * @param dataset Dataset a validar
 * @return true si es valido
 */
bool validarDataset(const Dataset& dataset);

/**
 * Calcula estadisticas completas de un dataset
 *
 * @param X Matriz de caracteristicas
 * @param y Vector de etiquetas
 * @return Estructura con estadisticas
 */
DatasetStats calcularEstadisticas(const std::vector<std::vector<AudioSample>>& X,
    const std::vector<int>& y);

/**
 * Calcula estadisticas (version con struct)
 *
 * @param dataset Dataset a analizar
 * @return Estructura con estadisticas
 */
DatasetStats calcularEstadisticas(const Dataset& dataset);

/**
 * Muestra estadisticas de un dataset en formato legible
 *
 * @param stats Estadisticas calculadas
 * @param nombre Nombre descriptivo del dataset
 */
void mostrarEstadisticas(const DatasetStats& stats, const std::string& nombre);

/**
 * Crea mapeo de speaker IDs a indices secuenciales
 * Util para convertir IDs arbitrarios a [0, 1, 2, ...]
 *
 * @param y Vector de etiquetas con IDs originales
 * @return SpeakerMapping con mapeos bidireccionales
 *
 * Ejemplo:
 *   y = [1001, 1001, 1005, 1003, 1005]
 *   mapping = crearMapeoSpeakers(y)
 *   // mapping.speaker_to_index = {1001->0, 1003->1, 1005->2}
 *   // mapping.index_to_speaker = {0->1001, 1->1003, 2->1005}
 */
SpeakerMapping crearMapeoSpeakers(const std::vector<int>& y);

/**
 * Aplica mapeo de speakers a las etiquetas
 * Convierte IDs originales a indices secuenciales
 *
 * @param y Vector de etiquetas con IDs originales (modificado in-place)
 * @param mapping Mapeo de speakers
 *
 * Ejemplo:
 *   y = [1001, 1003, 1001]
 *   aplicarMapeoSpeakers(y, mapping)
 *   // Ahora: y = [0, 1, 0]
 */
void aplicarMapeoSpeakers(std::vector<int>& y, const SpeakerMapping& mapping);

/**
 * Guarda mapeo de speakers en archivo de texto
 * Formato: speaker_id,index
 *
 * @param ruta Ruta del archivo de salida
 * @param mapping Mapeo a guardar
 * @return true si se guardo correctamente
 */
bool guardarMapeoSpeakers(const std::string& ruta, const SpeakerMapping& mapping);

/**
 * Carga mapeo de speakers desde archivo de texto
 *
 * @param ruta Ruta del archivo
 * @return SpeakerMapping cargado (vacio si error)
 */
SpeakerMapping cargarMapeoSpeakers(const std::string& ruta);

/**
 * Verifica compatibilidad entre dos datasets
 * Util para validar train/test antes de entrenar
 *
 * @param train Dataset de entrenamiento
 * @param test Dataset de prueba
 * @return true si son compatibles
 *
 * Validaciones:
 * - Misma dimension de caracteristicas
 * - Test no tiene clases no vistas en train
 */
bool verificarCompatibilidad(const Dataset& train, const Dataset& test);

/**
 * Fusiona multiples datasets en uno solo
 * Util para combinar datos de multiples fuentes
 *
 * @param datasets Vector de datasets a fusionar
 * @return Dataset fusionado
 */
Dataset fusionarDatasets(const std::vector<Dataset>& datasets);

/**
 * Filtra dataset por clases especificas
 *
 * @param dataset Dataset original
 * @param clases_incluir Vector con IDs de clases a mantener
 * @return Dataset filtrado
 */
Dataset filtrarPorClases(const Dataset& dataset,
    const std::vector<int>& clases_incluir);

/**
 * Crea un subset aleatorio del dataset
 *
 * @param dataset Dataset original
 * @param n_samples Numero de muestras a extraer
 * @param seed Semilla para reproducibilidad
 * @return Subset del dataset
 */
Dataset crearSubset(const Dataset& dataset, size_t n_samples, unsigned int seed);

#endif // DATASET_H