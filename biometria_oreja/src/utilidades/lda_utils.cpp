#include "utilidades/lda_utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// LDA (Linear Discriminant Analysis) - Fisher Linear Discriminant
// ============================================================================
// Pipeline: PCA -> LDA -> SVM
// LDA maximiza la separación entre clases (Sb) minimizando varianza intra-clase (Sw)
// Resultado: hasta (numClases - 1) componentes discriminativas

static void multiplicarMatrizVector(
    const std::vector<std::vector<double>>& M,
    const std::vector<double>& v,
    std::vector<double>& resultado
) {
    size_t rows = M.size();
    size_t cols = M[0].size();
    resultado.assign(rows, 0.0);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += M[i][j] * v[j];
        }
        resultado[i] = sum;
    }
}

static double normaVector(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

static void normalizarVector(std::vector<double>& v) {
    double norma = normaVector(v);
    if (norma > 1e-10) {
        for (double& x : v) x /= norma;
    }
}

// Power iteration para encontrar eigenvector dominante de una matriz
static std::vector<double> powerIteration(
    const std::vector<std::vector<double>>& M,
    int maxIter = 100,
    double tol = 1e-6
) {
    size_t n = M.size();
    std::vector<double> v(n, 1.0 / std::sqrt((double)n));
    std::vector<double> Mv(n);

    for (int iter = 0; iter < maxIter; ++iter) {
        multiplicarMatrizVector(M, v, Mv);
        double norma = normaVector(Mv);
        if (norma < 1e-10) break;

        double diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double nuevo = Mv[i] / norma;
            diff += (nuevo - v[i]) * (nuevo - v[i]);
            v[i] = nuevo;
        }

        if (std::sqrt(diff) < tol) break;
    }

    return v;
}

// Deflación: remueve la componente del eigenvector encontrado
static void deflacionar(
    std::vector<std::vector<double>>& M,
    const std::vector<double>& eigenvec,
    double eigenval
) {
    size_t n = M.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            M[i][j] -= eigenval * eigenvec[i] * eigenvec[j];
        }
    }
}

ModeloLDA entrenarLDA(
    const std::vector<std::vector<double>>& datos,
    const std::vector<int>& etiquetas,
    int numComponentes
) {
    ModeloLDA modelo;

    if (datos.empty()) {
        std::cerr << "[LDA] Error: datos vacios\n";
        return modelo;
    }

    size_t n = datos.size();      // Numero de muestras
    size_t d = datos[0].size();   // Dimensiones (ya reducidas por PCA)

    // Encontrar clases unicas
    std::unordered_set<int> clasesSet(etiquetas.begin(), etiquetas.end());
    std::vector<int> clases(clasesSet.begin(), clasesSet.end());
    std::sort(clases.begin(), clases.end());
    int numClases = static_cast<int>(clases.size());
    modelo.numClases = numClases;

    // LDA puede tener maximo (numClases - 1) componentes
    int maxComp = numClases - 1;
    if (numComponentes <= 0 || numComponentes > maxComp) {
        numComponentes = maxComp;
    }

    std::cout << "[LDA] Entrenando con " << n << " muestras, " << d << " dims, "
              << numClases << " clases\n";
    std::cout << "[LDA] Extrayendo " << numComponentes << " componentes discriminativas\n";

    // 1. Calcular media global
    modelo.mediaGlobal.assign(d, 0.0);
    for (const auto& muestra : datos) {
        for (size_t j = 0; j < d; ++j) {
            modelo.mediaGlobal[j] += muestra[j];
        }
    }
    for (size_t j = 0; j < d; ++j) {
        modelo.mediaGlobal[j] /= n;
    }

    // 2. Agrupar muestras por clase y calcular medias de clase
    std::unordered_map<int, std::vector<size_t>> indicesPorClase;
    for (size_t i = 0; i < n; ++i) {
        indicesPorClase[etiquetas[i]].push_back(i);
    }

    std::unordered_map<int, std::vector<double>> mediasClase;
    for (int c : clases) {
        mediasClase[c].assign(d, 0.0);
        for (size_t idx : indicesPorClase[c]) {
            for (size_t j = 0; j < d; ++j) {
                mediasClase[c][j] += datos[idx][j];
            }
        }
        size_t nc = indicesPorClase[c].size();
        for (size_t j = 0; j < d; ++j) {
            mediasClase[c][j] /= nc;
        }
    }

    // 3. Calcular matriz de dispersión entre clases (Sb)
    std::cout << "[LDA] Calculando matriz Sb (between-class scatter)...\n";
    std::vector<std::vector<double>> Sb(d, std::vector<double>(d, 0.0));

    for (int c : clases) {
        size_t nc = indicesPorClase[c].size();
        std::vector<double> diff(d);
        for (size_t j = 0; j < d; ++j) {
            diff[j] = mediasClase[c][j] - modelo.mediaGlobal[j];
        }

        // Sb += nc * (media_c - media_global) * (media_c - media_global)^T
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                Sb[i][j] += nc * diff[i] * diff[j];
            }
        }
    }

    // 4. Calcular matriz de dispersión intra-clase (Sw)
    std::cout << "[LDA] Calculando matriz Sw (within-class scatter)...\n";
    std::vector<std::vector<double>> Sw(d, std::vector<double>(d, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<std::vector<double>> Sw_local(d, std::vector<double>(d, 0.0));

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (size_t i = 0; i < n; ++i) {
            int c = etiquetas[i];
            std::vector<double> diff(d);
            for (size_t j = 0; j < d; ++j) {
                diff[j] = datos[i][j] - mediasClase[c][j];
            }

            // Sw += (x - media_clase) * (x - media_clase)^T
            for (size_t j = 0; j < d; ++j) {
                for (size_t k = 0; k < d; ++k) {
                    Sw_local[j][k] += diff[j] * diff[k];
                }
            }
        }

        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (size_t i = 0; i < d; ++i) {
                for (size_t j = 0; j < d; ++j) {
                    Sw[i][j] += Sw_local[i][j];
                }
            }
        }
    }

    // 5. Regularización de Sw para estabilidad numérica
    double traza = 0.0;
    for (size_t i = 0; i < d; ++i) {
        traza += Sw[i][i];
    }
    double alpha = 0.001 * traza / d;  // 0.1% de regularización
    for (size_t i = 0; i < d; ++i) {
        Sw[i][i] += alpha;
    }

    // 6. Resolver Sw^(-1) * Sb usando Cholesky + substitución
    // Simplificación: usamos inversión directa con regularización
    std::cout << "[LDA] Calculando Sw^(-1) * Sb (con regularizacion)...\n";

    // Calcular inversa de Sw usando Gauss-Jordan con pivoteo
    std::vector<std::vector<double>> SwInv(d, std::vector<double>(d, 0.0));
    std::vector<std::vector<double>> SwAug(d, std::vector<double>(2*d, 0.0));

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            SwAug[i][j] = Sw[i][j];
        }
        SwAug[i][d + i] = 1.0;  // Matriz identidad aumentada
    }

    // Gauss-Jordan
    for (size_t i = 0; i < d; ++i) {
        // Pivoteo parcial
        size_t maxRow = i;
        for (size_t k = i + 1; k < d; ++k) {
            if (std::abs(SwAug[k][i]) > std::abs(SwAug[maxRow][i])) {
                maxRow = k;
            }
        }
        std::swap(SwAug[i], SwAug[maxRow]);

        double pivot = SwAug[i][i];
        if (std::abs(pivot) < 1e-10) {
            pivot = 1e-10;  // Evitar división por cero
        }

        for (size_t j = 0; j < 2*d; ++j) {
            SwAug[i][j] /= pivot;
        }

        for (size_t k = 0; k < d; ++k) {
            if (k != i) {
                double factor = SwAug[k][i];
                for (size_t j = 0; j < 2*d; ++j) {
                    SwAug[k][j] -= factor * SwAug[i][j];
                }
            }
        }
    }

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            SwInv[i][j] = SwAug[i][d + j];
        }
    }

    // Calcular M = Sw^(-1) * Sb
    std::vector<std::vector<double>> M(d, std::vector<double>(d, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < d; ++k) {
                sum += SwInv[i][k] * Sb[k][j];
            }
            M[i][j] = sum;
        }
    }

    // 7. Extraer eigenvectores de M usando power iteration + deflación
    std::cout << "[LDA] Extrayendo " << numComponentes << " eigenvectores...\n";
    modelo.componentes.reserve(numComponentes);

    for (int comp = 0; comp < numComponentes; ++comp) {
        std::vector<double> eigenvec = powerIteration(M, 200, 1e-8);

        // Calcular eigenvalue aproximado
        std::vector<double> Mv;
        multiplicarMatrizVector(M, eigenvec, Mv);
        double eigenval = 0.0;
        for (size_t i = 0; i < d; ++i) {
            eigenval += eigenvec[i] * Mv[i];
        }

        modelo.componentes.push_back(eigenvec);

        // Deflacionar para encontrar siguiente eigenvector
        deflacionar(M, eigenvec, eigenval);

        if ((comp + 1) % 20 == 0 || comp == numComponentes - 1) {
            std::cout << "  [LDA] Componente " << (comp + 1) << "/" << numComponentes << "\n";
        }
    }

    std::cout << "[LDA] Completado: " << modelo.componentes.size() << " componentes extraidas\n";

    return modelo;
}

std::vector<std::vector<double>> aplicarLDAConModelo(
    const std::vector<std::vector<double>>& datos,
    const ModeloLDA& modelo
) {
    std::vector<std::vector<double>> resultado;

    if (datos.empty() || modelo.componentes.empty()) {
        return resultado;
    }

    size_t n = datos.size();
    size_t numComp = modelo.componentes.size();
    size_t d = datos[0].size();

    resultado.resize(n, std::vector<double>(numComp, 0.0));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < n; ++i) {
        // Centrar dato respecto a media global
        std::vector<double> centrado(d);
        for (size_t j = 0; j < d; ++j) {
            centrado[j] = datos[i][j] - modelo.mediaGlobal[j];
        }

        // Proyectar en cada componente LDA
        for (size_t c = 0; c < numComp; ++c) {
            double proj = 0.0;
            for (size_t j = 0; j < d; ++j) {
                proj += centrado[j] * modelo.componentes[c][j];
            }
            resultado[i][c] = proj;
        }
    }

    return resultado;
}

bool guardarModeloLDA(const std::string& ruta, const ModeloLDA& modelo) {
    std::ofstream file(ruta);
    if (!file.is_open()) {
        std::cerr << "[LDA] Error: no se pudo abrir " << ruta << " para escritura\n";
        return false;
    }

    // Linea 1: numClases, numComponentes, dimensiones
    size_t numComp = modelo.componentes.size();
    size_t dims = modelo.componentes.empty() ? 0 : modelo.componentes[0].size();
    file << modelo.numClases << ";" << numComp << ";" << dims << "\n";

    // Linea 2: media global
    for (size_t i = 0; i < modelo.mediaGlobal.size(); ++i) {
        if (i > 0) file << ";";
        file << modelo.mediaGlobal[i];
    }
    file << "\n";

    // Lineas 3+: componentes
    for (const auto& comp : modelo.componentes) {
        for (size_t i = 0; i < comp.size(); ++i) {
            if (i > 0) file << ";";
            file << comp[i];
        }
        file << "\n";
    }

    file.close();
    std::cerr << "[LDA] Modelo guardado en: " << ruta << "\n";
    return true;
}

ModeloLDA cargarModeloLDA(const std::string& ruta) {
    ModeloLDA modelo;
    std::ifstream file(ruta);

    if (!file.is_open()) {
        std::cerr << "[LDA] Error: no se pudo abrir " << ruta << "\n";
        return modelo;
    }

    std::string linea;

    // Linea 1: metadata
    if (!std::getline(file, linea)) return modelo;
    std::istringstream iss1(linea);
    std::string token;
    std::getline(iss1, token, ';'); modelo.numClases = std::stoi(token);
    std::getline(iss1, token, ';'); int numComp = std::stoi(token);
    std::getline(iss1, token, ';'); int dims = std::stoi(token);

    // Linea 2: media global
    if (!std::getline(file, linea)) return modelo;
    std::istringstream iss2(linea);
    modelo.mediaGlobal.reserve(dims);
    while (std::getline(iss2, token, ';')) {
        modelo.mediaGlobal.push_back(std::stod(token));
    }

    // Lineas 3+: componentes
    modelo.componentes.reserve(numComp);
    while (std::getline(file, linea)) {
        std::istringstream issComp(linea);
        std::vector<double> comp;
        comp.reserve(dims);
        while (std::getline(issComp, token, ';')) {
            comp.push_back(std::stod(token));
        }
        if (!comp.empty()) {
            modelo.componentes.push_back(comp);
        }
    }

    file.close();
    std::cerr << "[LDA] Modelo cargado: " << modelo.componentes.size()
              << " componentes, " << dims << " dims\n";
    return modelo;
}
