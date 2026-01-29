#include "utilidades/pca_utils.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <omp.h>

// === Helpers internos ===
static void normalizar(std::vector<double>& v) {
    double norma = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    if (norma > 1e-12) {
        for (double& x : v) x /= norma;
    }
}

static double productoPunto(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static std::vector<double> multiplicarMatrizVector(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v
) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    std::vector<double> resultado(rows, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double* row = A[i].data();
        const double* vec = v.data();
        for (size_t j = 0; j < cols; ++j) {
            sum += row[j] * vec[j];
        }
        resultado[i] = sum;
    }
    return resultado;
}

static std::vector<std::vector<double>> obtenerComponentesPrincipales_HELPER(
    const std::vector<std::vector<double>>& cov,
    int k
) {
    size_t n = cov.size();
    std::vector<std::vector<double>> componentes;
    componentes.reserve((size_t)k);

    std::cout << "  [PCA] Extrayendo " << k << " componentes principales (con OpenMP)..." << std::endl;

    for (int comp = 0; comp < k; ++comp) {
        if (comp % 5 == 0 || comp == k - 1) {
            int pct = (int)(100.0 * comp / k);
            std::cout << "\r  [PCA] Componente " << comp << "/" << k << " (" << pct << "%)" << std::flush;
        }
        
        std::vector<double> b_k(n, 1.0);
        normalizar(b_k);

        // RESTAURADO: 1000 iteraciones para precisión (era 100, pero causó -15% test accuracy)
        for (int iter = 0; iter < 1000; ++iter) {
            std::vector<double> b_k1 = multiplicarMatrizVector(cov, b_k);

            // Ortogonalizar contra vectores anteriores
            for (const auto& v : componentes) {
                double proy = productoPunto(b_k1, v);
                for (size_t i = 0; i < n; ++i)
                    b_k1[i] -= proy * v[i];
            }

            normalizar(b_k1);

            double diff = 0.0;
            for (size_t i = 0; i < n; ++i)
                diff += std::abs(b_k[i] - b_k1[i]);

            if (diff < 1e-6) break;
            b_k = std::move(b_k1);
        }

        componentes.push_back(std::move(b_k));
    }
    std::cout << "\r  [PCA] Componente " << k << "/" << k << " (100%) - Completado." << std::endl;

    return componentes;
}

static std::vector<std::vector<double>> centrarDatos_HELPER(
    const std::vector<std::vector<double>>& datos,
    std::vector<double>& media
) {
    size_t m = datos.size();
    size_t n = datos[0].size();
    media.assign(n, 0.0);

    for (const auto& fila : datos)
        for (size_t j = 0; j < n; ++j)
            media[j] += fila[j];

    for (size_t j = 0; j < n; ++j)
        media[j] /= (m > 0 ? (double)m : 1.0);

    std::vector<std::vector<double>> centrados(m, std::vector<double>(n));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            centrados[i][j] = datos[i][j] - media[j];

    return centrados;
}

static std::vector<std::vector<double>> calcularCovarianza_HELPER(
    const std::vector<std::vector<double>>& datos
) {
    size_t m = datos.size();
    size_t n = datos[0].size();
    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));

    std::cout << "  [PCA] Calculando covarianza con OpenMP..." << std::flush;

    // Paralelizar por filas de la matriz de covarianza
    #pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = j; k < n; ++k) {  // Solo triángulo superior (simétrica)
            double sum = 0.0;
            for (size_t i = 0; i < m; ++i) {
                sum += datos[i][j] * datos[i][k];
            }
            cov[j][k] = sum;
            cov[k][j] = sum;  // Simetría
        }
    }
    std::cout << " completado." << std::endl;

    double denom = (m > 1) ? (double)(m - 1) : 1.0;

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            cov[j][k] /= denom;
        }
    }

    return cov;
}

// Parse robusto (tipo Ubuntu viejo)
static std::vector<double> parseLineaCSVNumerica(const std::string& linea) {
    std::vector<double> valores;
    std::stringstream ss(linea);
    std::string val;

    while (std::getline(ss, val, ',')) {
        // trim espacios + \r\n
        auto first = val.find_first_not_of(" \t\r\n");
        auto last  = val.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) continue;

        std::string limpio = val.substr(first, last - first + 1);
        if (limpio.empty()) continue;

        // validar token numérico simple
        bool pareceNumero = true;
        for (unsigned char c : limpio) {
            if (!((c >= '0' && c <= '9') ||
                  c == '+' || c == '-' ||
                  c == '.' || c == 'e' || c == 'E')) {
                pareceNumero = false;
                break;
            }
        }
        if (!pareceNumero) continue;

        try {
            valores.push_back(std::stod(limpio));
        } catch (...) {
            // ignorar token inválido
        }
    }

    return valores;
}

ModeloPCA entrenarPCA(const std::vector<std::vector<double>>& datos, int numComponentes) {
    ModeloPCA modelo;
    if (datos.empty() || datos[0].empty() || numComponentes <= 0) return modelo;

    std::cout << "[PCA] Centrando datos (" << datos.size() << " x " << datos[0].size() << ")..." << std::endl;
    std::vector<double> media;
    auto datosCentrados = centrarDatos_HELPER(datos, media);
    
    std::cout << "[PCA] Calculando matriz de covarianza " << datos[0].size() << "x" << datos[0].size() 
              << " (esto puede tomar 5-10 minutos)..." << std::endl;
    auto cov = calcularCovarianza_HELPER(datosCentrados);
    
    std::cout << "[PCA] Calculando eigenvalores/eigenvectores..." << std::endl;
    // evita pedir más componentes que dimensión
    int dim = (int)cov.size();
    int k = std::min(numComponentes, dim);

    auto componentes = obtenerComponentesPrincipales_HELPER(cov, k);

    modelo.medias = std::move(media);
    modelo.componentes = std::move(componentes);
    std::cout << "[PCA] Completado: " << k << " componentes extraídos." << std::endl;
    return modelo;
}

std::vector<std::vector<double>> aplicarPCAConModelo(
    const std::vector<std::vector<double>>& datos,
    const ModeloPCA& modelo
) {
    std::vector<std::vector<double>> resultado;
    if (modelo.medias.empty() || modelo.componentes.empty()) return resultado;

    resultado.reserve(datos.size());

    for (const auto& fila : datos) {
        if (fila.size() != modelo.medias.size()) {
            std::cerr << "Dimensiones incompatibles: entrada = " << fila.size()
                      << ", PCA espera = " << modelo.medias.size() << "\n";
            continue;
        }

        std::vector<double> centrado(fila.size());
        for (size_t j = 0; j < fila.size(); ++j)
            centrado[j] = fila[j] - modelo.medias[j];

        std::vector<double> reducida(modelo.componentes.size(), 0.0);
        for (size_t i = 0; i < modelo.componentes.size(); ++i)
            for (size_t j = 0; j < centrado.size(); ++j)
                reducida[i] += modelo.componentes[i][j] * centrado[j];

        resultado.push_back(std::move(reducida));
    }

    return resultado;
}

bool aplicarPCADesdeModelo(
    const std::string& rutaModelo,
    const std::vector<std::vector<double>>& datosEntrada,
    std::vector<std::vector<double>>& salidaPCA
) {
    ModeloPCA modelo = cargarModeloPCA(rutaModelo);
    if (modelo.medias.empty() || modelo.componentes.empty()) return false;
    salidaPCA = aplicarPCAConModelo(datosEntrada, modelo);
    return true;
}

bool guardarModeloPCA(const std::string& ruta, const ModeloPCA& modelo) {
    const auto parent = std::filesystem::path(ruta).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);

    std::ofstream archivo(ruta);
    if (!archivo.is_open()) return false;

    // medias
    for (double m : modelo.medias) archivo << m << ",";
    archivo << "\n";

    // componentes
    for (const auto& fila : modelo.componentes) {
        for (double val : fila) archivo << val << ",";
        archivo << "\n";
    }

    return true;
}

ModeloPCA cargarModeloPCA(const std::string& ruta) {
    ModeloPCA modelo;
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "[cargarModeloPCA] No se pudo abrir: " << ruta << "\n";
        return modelo;
    }

    std::string linea;

    // medias (primera línea)
    if (std::getline(archivo, linea)) {
        modelo.medias = parseLineaCSVNumerica(linea);
        if (modelo.medias.empty()) {
            std::cerr << "[cargarModeloPCA] Advertencia: medias vacías tras parsear "
                      << ruta << "\n";
        }
    }

    // componentes (resto)
    while (std::getline(archivo, linea)) {
        auto fila = parseLineaCSVNumerica(linea);
        if (!fila.empty()) modelo.componentes.push_back(std::move(fila));
    }

    if (modelo.componentes.empty()) {
        std::cerr << "[cargarModeloPCA] Advertencia: sin componentes válidos en "
                  << ruta << "\n";
    }

    return modelo;
}
