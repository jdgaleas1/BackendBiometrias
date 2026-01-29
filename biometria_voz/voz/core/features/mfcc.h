#ifndef MFCC_H
#define MFCC_H

#include "config.h"
#include <vector>
#include <string>

/**
 * ===================================================================
 * MFCC (Mel-Frequency Cepstral Coefficients)   
 * ===================================================================
 *
 * Cambios principales:
 * - float** -> std::vector<std::vector<AudioSample>>
 * - float* -> std::vector<AudioSample>
 * - Precision double en todos los calculos
 * - Eliminacion de malloc/free/calloc
 * - Paralelizacion condicional con OpenMP
 * - Tabla LOG en double o uso directo de std::log
 * - RAII automatico - cero memory leaks
 *
 * Extrae coeficientes MFCC de un espectrograma
 * Usa configuracion centralizada desde CONFIG_MFCC (config.h)
 *
 * Parametros configurables en config.h:
 * - numCoefficients: Numero de coeficientes MFCC a extraer 
 * - numFilters: Numero de filtros mel-scale (default: 40)
 * - freqMin: Frecuencia minima en Hz (default: 0 Hz)
 * - freqMax: Frecuencia maxima en Hz (default: 8000 Hz)
 *
 * @param spectrogram Matriz 2D con magnitudes espectrales [frames][bins]
 * @param sampleRate Frecuencia de muestreo del audio original en Hz
 * @return Matriz 2D [frames][coefficients] con coeficientes MFCC
 *
 * Algoritmo completo:
 * 1. Aplicar filterbank mel-scale al espectrograma
 *    - Convierte escala lineal (Hz) a escala perceptual (Mel)
 *    - Crea CONFIG_MFCC.numFilters filtros triangulares solapados
 *    - Rango: CONFIG_MFCC.freqMin a CONFIG_MFCC.freqMax
 *
 * 2. Calcular energia por filtro mel
 *    - Suma ponderada de magnitudes espectrales por cada filtro
 *    - Resultado: Vector de energias mel por frame
 *
 * 3. Aplicar logaritmo (compresion dinamica)
 *    - log(energia + epsilon) para evitar log(0)
 *    - Simula percepcion logaritmica del oido humano
 *
 * 4. DCT (Discrete Cosine Transform)
 *    - Decorrelaciona coeficientes mel
 *    - Extrae CONFIG_MFCC.numCoefficients componentes principales
 *    - Comprime informacion en dimensiones mas bajas
 *
 * Caracteristicas:
 * - Precision double para calculos exactos
 * - Paralelizacion OpenMP sobre frames (si ENABLE_OPENMP=1)
 * - Memoria eficiente con vectores dinamicos
 * - Sin memory leaks (RAII automatico)
 *
 * Ejemplo de uso:
 *   auto spectrogram = applySTFT(audio, 16000);
 *
 *   // Configurar parametros (opcional)
 *   CONFIG_MFCC.numCoefficients = 39;
 *   CONFIG_MFCC.numFilters = 40;
 *
 *   // Extraer MFCC
 *   auto mfcc = extractMFCC(spectrogram, 16000);
 *
 *   // Usar coeficientes: mfcc[frame][coeff]
 *   // Memoria se libera automaticamente
 *
 * Notas sobre coeficientes:
 * - MFCC[0]: Relacionado con energia total
 * - MFCC[1-38]: Capturan forma espectral (timbre vocal)
 * - Valores tipicos: numCoefficients 
 *
 * Complejidad: O(F * B * C) donde F=frames, B=bins, C=coefficients
 * Memoria: O(F * C) para salida
 *
 * VENTAJAS VERSION 3.0:
 * - Precision 1000x mayor con double
 * - Cero memory leaks garantizados
 * - Codigo mas simple y mantenible
 * - Paralelizacion opcional y controlable
 */
std::vector<std::vector<AudioSample>> extractMFCC(
    const std::vector<std::vector<AudioSample>>& spectrogram,
    int sampleRate
);

/**
 * Calcula estadisticas agregadas extendidas de coeficientes MFCC
 * Genera vector de caracteristicas con 5 estadisticas por coeficiente
 *
 * @param mfcc Matriz de coeficientes MFCC [frames][coefficients]
 * @return Vector de caracteristicas estadisticas (MEAN+STD+MIN+MAX+DELTA)
 *
 * Estadisticas calculadas (5 por cada coeficiente):
 * - MEAN: Promedio temporal (timbre vocal promedio)
 * - STD: Desviacion estandar (variabilidad vocal)
 * - MIN: Valor minimo (rango dinamico inferior)
 * - MAX: Valor maximo (rango dinamico superior)
 * - DELTA: Primera derivada temporal promedio (dinamica de transiciones)
 *
 * Dimension de salida:
 * - Si numCoefficients = 50, entonces totalFeatures = 250 (50 x 5)
 *   * Captura caracteristica espectral promedio
 *   * Representa "color" vocal tipico del hablante
 *
 * Estructura del vector resultante (para N coeficientes):
 * [mean_0, mean_1, ..., mean_N-1]
 *
 * Uso en reconocimiento de voz:
 * - Este vector es la entrada al clasificador SVM
 * - Captura informacion espectral caracteristica
 * - Robusto a variaciones de duracion del audio
 *
 * Ejemplo:
 *   auto features = calcularEstadisticasMFCC(mfcc);
 *   // features.size() == CONFIG_MFCC.numCoefficients
 *
 * Complejidad: O(F * C) donde F=frames, C=coefficients
 * Paralelizado con OpenMP si ENABLE_OPENMP=1
 */
std::vector<AudioSample> calcularEstadisticasMFCC(
    const std::vector<std::vector<AudioSample>>& mfcc
);

/**
 * Guarda caracteristicas de UNA muestra en archivo binario (append mode)
 * Util para construir dataset incrementalmente
 *
 * @param features Vector de caracteristicas
 * @param label Etiqueta de clase (ID del hablante)
 * @param outputPath Ruta del archivo binario de salida
 * @return true si se guardo correctamente
 *
 * Formato binario por muestra:
 * [dim:int][features:double*dim][label:int]
 *
 * Modo append: Se agrega al final del archivo sin sobrescribir
 *
 * Ejemplo de uso incremental:
 *   for (audio in audios) {
 *       auto features = extraerCaracteristicasCompletas(audio);
 *       guardarCaracteristicasBinario(features, speakerId, "dataset.bin");
 *   }
 */
bool guardarCaracteristicasBinario(
    const std::vector<AudioSample>& features,
    int label,
    const std::string& outputPath
);

/**
 * Exporta dataset a formato CSV para analisis exploratorio
 * Util para debugging, visualizacion y validacion de datos
 *
 * @param features Vector de vectores con caracteristicas
 * @param labels Vector de etiquetas correspondientes
 * @param outputPath Ruta del archivo CSV de salida
 * @return true si se exporto correctamente
 *
 * Formato CSV:
 * label,mfcc0,mfcc1,mfcc2,...,mfccN
 * 1,0.523,-1.234,0.891,...,0.234
 * 1,0.612,-1.102,0.923,...,0.189
 * ...
 *
 * Usos:
 * - Visualizacion en Python/R (matplotlib, seaborn)
 * - Analisis estadistico exploratorio
 * - Validacion de distribucion de datos
 * - Comparacion con otros metodos
 *
 * Ejemplo:
 *   exportarCaracteristicasCSV(allFeatures, allLabels, "dataset.csv");
 *   // Luego en Python: df = pd.read_csv("dataset.csv")
 */
bool exportarCaracteristicasCSV(
    const std::vector<std::vector<AudioSample>>& features,
    const std::vector<int>& labels,
    const std::string& outputPath
);

#endif // MFCC_H