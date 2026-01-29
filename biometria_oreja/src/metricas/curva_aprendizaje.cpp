/* #include "metricas/curva_aprendizaje.h"

#include "svm/svm_entrenamiento.h"
#include "svm/svm_prediccion.h"
#include "metricas/svm_metricas.h"
#include "utilidades/dividir_dataset.h"
#include "svm/cargar_csv.h"

#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

static void asegurarDirectorio(const std::string& rutaArchivo) {
    fs::create_directories(fs::path(rutaArchivo).parent_path());
}

static void escribirResumen(const std::string& rutaBase,
    const std::vector<int>& ks,
    const std::vector<double>& acc_mean,
    const std::vector<double>& acc_std,
    const std::vector<double>& f1_mean,
    const std::vector<double>& f1_std,
    const std::vector<double>& bacc_mean,
    const std::vector<double>& bacc_std,
    const std::vector<double>& mcc_mean,
    const std::vector<double>& mcc_std) {
    const std::string ruta = fs::path(rutaBase).replace_filename(
        fs::path(rutaBase).stem().string() + "_resumen.csv"
    ).string();
    std::ofstream out(ruta);
    if (!out) return;
    out << "num_clases,acc_mean,acc_std,f1_macro_mean,f1_macro_std,balanced_acc_mean,balanced_acc_std,mcc_mean,mcc_std\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < ks.size(); ++i) {
        out << ks[i] << ","
            << acc_mean[i] << "," << acc_std[i] << ","
            << f1_mean[i] << "," << f1_std[i] << ","
            << bacc_mean[i] << "," << bacc_std[i] << ","
            << mcc_mean[i] << "," << mcc_std[i] << "\n";
    }
}

void generarCurvaAprendizaje(const std::string& rutaCSV,
    const std::vector<int>& clases_a_usar,
    int repeticiones,
    const std::string& rutaSalidaCSV,
    unsigned int seed_base) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    if (!cargarCSV(rutaCSV, X, y, ';')) {
        std::cerr << "No se pudo cargar el archivo " << rutaCSV << "\n";
        return;
    }

    // Agrupar por clase
    std::map<int, std::vector<const std::vector<double>*>> porClasePtr;
    for (size_t i = 0; i < y.size(); ++i) porClasePtr[y[i]].push_back(&X[i]);

    asegurarDirectorio(rutaSalidaCSV);
    std::ofstream salida(rutaSalidaCSV);
    if (!salida) {
        std::cerr << "No se pudo abrir salida: " << rutaSalidaCSV << "\n";
        return;
    }
    salida << "num_clases,rep,accuracy,f1_macro,balanced_acc,mcc\n";
    salida << std::fixed << std::setprecision(6);

    // Acumuladores para resumen por K
    std::vector<double> acc_mean, acc_std, f1_mean, f1_std, bacc_mean, bacc_std, mcc_mean, mcc_std;
    acc_mean.reserve(clases_a_usar.size());
    acc_std.reserve(clases_a_usar.size());
    f1_mean.reserve(clases_a_usar.size());
    f1_std.reserve(clases_a_usar.size());
    bacc_mean.reserve(clases_a_usar.size());
    bacc_std.reserve(clases_a_usar.size());
    mcc_mean.reserve(clases_a_usar.size());
    mcc_std.reserve(clases_a_usar.size());

    for (int num_clases : clases_a_usar) {
        std::vector<double> accs, f1s, baccs, mccs;
        accs.reserve(repeticiones);
        f1s.reserve(repeticiones);
        baccs.reserve(repeticiones);
        mccs.reserve(repeticiones);

        // Armar lista de clases con mínimo soporte (>= 42 si vas a 30 train + 12 test)
        std::vector<int> clases_disponibles;
        for (const auto& kv : porClasePtr) {
            if ((int)kv.second.size() >= 42) clases_disponibles.push_back(kv.first);
        }
        if ((int)clases_disponibles.size() < num_clases) {
            std::cerr << "No hay suficientes clases con soporte para K=" << num_clases << "\n";
            acc_mean.push_back(0); acc_std.push_back(0);
            f1_mean.push_back(0);  f1_std.push_back(0);
            bacc_mean.push_back(0); bacc_std.push_back(0);
            mcc_mean.push_back(0);  mcc_std.push_back(0);
            continue;
        }

        for (int rep = 0; rep < repeticiones; ++rep) {
            std::mt19937 gen(seed_base + rep + num_clases * 9973);
            auto clases = clases_disponibles;
            std::shuffle(clases.begin(), clases.end(), gen);
            clases.resize(num_clases);

            // Re-etiquetar a 0..K-1 y construir X_sel/y_sel
            std::map<int, int> remap;
            for (int k = 0; k < num_clases; ++k) remap[clases[k]] = k;

            std::vector<std::vector<double>> X_sel;
            std::vector<int> y_sel;
            for (int c : clases) {
                for (const auto* ptr : porClasePtr[c]) {
                    X_sel.push_back(*ptr);
                    y_sel.push_back(remap[c]);
                }
            }

            // Split estratificado (mantén tus parámetros 30/12 o cámbialos aquí)
            std::vector<std::vector<double>> X_train, X_test;
            std::vector<int> y_train, y_test;
            dividirEstratificado(X_sel, y_sel, X_train, y_train, X_test, y_test, 30, 12);

            // Entrenar y evaluar
            ModeloSVM modelo = entrenarSVMOVA(X_train, y_train, 0.05, 7000, 0.0001, 1e-4);
            std::vector<int> y_pred;
            y_pred.reserve(y_test.size());
            for (const auto& feat : X_test) y_pred.push_back(predecirPersona(feat, modelo));

            ResultadosMetricas m = calcularMetricasAvanzadas(y_test, y_pred, num_clases);
            accs.push_back(m.accuracy);
            f1s.push_back(m.f1_macro);
            baccs.push_back(m.balanced_accuracy);
            mccs.push_back(m.mcc);

            salida << num_clases << "," << rep << ","
                << m.accuracy << "," << m.f1_macro << ","
                << m.balanced_accuracy << "," << m.mcc << "\n";
        }

        auto media = [](const std::vector<double>& v) {
            if (v.empty()) return 0.0;
            double s = std::accumulate(v.begin(), v.end(), 0.0);
            return s / v.size();
            };
        auto stddev = [&](const std::vector<double>& v) {
            if (v.size() < 2) return 0.0;
            double mu = media(v), acc = 0.0;
            for (double x : v) { double d = x - mu; acc += d * d; }
            return std::sqrt(acc / (v.size() - 1));
            };

        acc_mean.push_back(media(accs));  acc_std.push_back(stddev(accs));
        f1_mean.push_back(media(f1s));    f1_std.push_back(stddev(f1s));
        bacc_mean.push_back(media(baccs)); bacc_std.push_back(stddev(baccs));
        mcc_mean.push_back(media(mccs));  mcc_std.push_back(stddev(mccs));
    }

    salida.close();
    escribirResumen(rutaSalidaCSV, clases_a_usar, acc_mean, acc_std, f1_mean, f1_std, bacc_mean, bacc_std, mcc_mean, mcc_std);
    std::cout << "Curva de aprendizaje exportada: " << rutaSalidaCSV << "\n";
}
 */