#include "admin/admin_report.h"

#include <sstream>
#include <iomanip>

static std::string vecSample10(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    int n = (int)std::min<size_t>(10, v.size());
    for (int i = 0; i < n; ++i) {
        oss << std::fixed << std::setprecision(3) << v[i];
        if (i + 1 < n) oss << ", ";
    }
    if ((int)v.size() > n) oss << ", ...";
    oss << "]";
    return oss.str();
}

void startupLogs(std::ofstream& log, const Ctx& ctx) {
    std::vector<std::string> lines;

    lines.push_back("Proyecto: Microservicio biometria de oreja (registro incremental SVM-OVA)");
    lines.push_back("WORK_DIR: " + ctx.workDir);
    lines.push_back("MODEL_DIR: " + ctx.modelDir);
    lines.push_back(std::string("AUDIT_MODE=") + (ctx.AUDIT_MODE ? "1" : "0") +
                    " | LOG_DETAIL=" + std::to_string(ctx.LOG_DETAIL) +
                    " | debug=" + (ctx.debug ? "1" : "0"));

    lines.push_back("Rutas:");
    lines.push_back("  - CSV: " + ctx.rutaCSV);
    lines.push_back("  - PCA: " + ctx.rutaModeloPCA);
    lines.push_back("  - SVM: " + ctx.rutaModeloSVM);
    lines.push_back("  - Holdout: " + ctx.holdoutCsv);

    lines.push_back("Config:");
    lines.push_back("  - PERF_DROP_THRESHOLD=" + std::to_string(ctx.PERF_DROP_THRESHOLD));
    lines.push_back("  - EVAL_PRINT_N=" + std::to_string(ctx.EVAL_PRINT_N));
    lines.push_back("  - QC_MIN_PASS=" + std::to_string(ctx.QC_MIN_PASS));
    lines.push_back("  - QC_ENFORCE=" + std::to_string(ctx.QC_ENFORCE));

    lines.push_back("QC Umbrales:");
    lines.push_back("  - mean=[" + std::to_string(ctx.QC.mean_min) + ".." + std::to_string(ctx.QC.mean_max) + "]");
    lines.push_back("  - std_min=" + std::to_string(ctx.QC.std_min) +
                    " | min_min=" + std::to_string(ctx.QC.min_min) +
                    " | max_max=" + std::to_string(ctx.QC.max_max));
    lines.push_back("  - pct_dark_max=" + std::to_string(ctx.QC.pct_dark_max) + "%" +
                    " | pct_bright_max=" + std::to_string(ctx.QC.pct_bright_max) + "%");
    lines.push_back("  - dark_thr=" + std::to_string(ctx.QC.dark_thr) +
                    " | bright_thr=" + std::to_string(ctx.QC.bright_thr));

    logBox(log, ctx.rid, ctx.LOG_DETAIL,
           "INICIO REGISTRO BIOMETRICO DE OREJA (AGREGAR_USUARIO_BIOMETRIA)",
           lines);
}

void logBloquePorImagen(std::ofstream& log, const Ctx& ctx,
                               int idxPlus1, int total,
                               const std::string& ruta,
                               const ImageReport& r) {
    if (ctx.LOG_DETAIL < 2) return;

    std::ostringstream oss;
    oss << "------------------------------------------------------------\n";
    oss << "RESUMEN (por imagen)\n";
    oss << "------------------------------------------------------------\n";
    oss << "IMG " << idxPlus1 << "/" << total << " | " << r.name << "\n";
    oss << "ruta=" << ruta << "\n";
    oss << "LOAD:     " << (r.loadOk ? "OK" : "FAIL") << " (" << r.ms_load << " ms)\n";
    oss << "QC:       " << (r.qcOk ? "PASS" : "FAIL");
    if (!r.qcOk && !r.qcReason.empty()) oss << " reason=" << r.qcReason;
    oss << "\n";
    oss << "PREPROC:  " << (r.preprocOk ? "OK" : "FAIL") << " (" << r.ms_preproc << " ms)\n";
    oss << "AUG:      count=" << r.augCount << "\n";
    oss << "FEATS:    vectores=" << r.featCount << " dims=" << r.dims
        << " (" << r.ms_feats << " ms)\n";

    if (!r.err.empty()) oss << "ERROR: " << r.err << "\n";
    oss << "\n";

    logRaw(log, ctx.rid, oss.str());
}

void logTablaQC(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                       const std::vector<ImageReport>& rep,
                       const QcThresholds& QC, int QC_MIN_PASS) {
    if (LOG_DETAIL < 2) return;
    
    logSection(log, rid, LOG_DETAIL, "TABLA RESUMEN: CONTROL DE CALIDAD (QC)");
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss.str(""); oss.clear(); oss << "  - mean:       [" << QC.mean_min << ", " << QC.mean_max << "]";
    logRawLine(log, rid, oss.str());

    oss.str(""); oss.clear(); oss << "  - std:        >= " << QC.std_min;
    logRawLine(log, rid, oss.str());

    oss.str(""); oss.clear(); oss << "  - pct_dark:   <= " << QC.pct_dark_max << "%";
    logRawLine(log, rid, oss.str());

    oss.str(""); oss.clear(); oss << "  - pct_bright: <= " << QC.pct_bright_max << "%";
    logRawLine(log, rid, oss.str());

    oss.str(""); oss.clear(); oss << "  - dark_thr=" << QC.dark_thr << "  bright_thr=" << QC.bright_thr;
    logRawLine(log, rid, oss.str());

    oss.str(""); oss.clear(); oss << "  - regla_global: min_pass = " << QC_MIN_PASS
                                << "/" << rep.size() << " imágenes";
    logRawLine(log, rid, oss.str());
    
    logRawLine(log, rid, "┌──────────┬──────┬──────┬──────┬─────────────────────┐");
    logRawLine(log, rid, "│ Imagen   │ QC   │ Mean │ Std  │ Razón               │");
    logRawLine(log, rid, "├──────────┼──────┼──────┼──────┼─────────────────────┤");
    
    int passCount = 0;
    for (size_t i = 0; i < rep.size(); ++i) {
        const auto& r = rep[i];
        if (r.qcOk) passCount++;
        
        std::ostringstream row;
        row << "│ " << std::setw(8) << std::left << ("img_" + std::to_string(i)) << " │ "
            << std::setw(4) << std::left << (r.qcOk ? "PASS" : "FAIL") << " │ "
            << std::setw(4) << std::fixed << std::setprecision(2) << r.mean << " │ "
            << std::setw(4) << std::fixed << std::setprecision(1) << r.std << " │ "
            << std::setw(19) << std::left << (r.qcOk ? "-" : r.qcReason) << " │";
        logRawLine(log, rid, row.str());
    }
    
    logRawLine(log, rid, "└──────────┴──────┴──────┴──────┴─────────────────────┘");
    logRawLine(log, rid, "");
    logRawLine(log, rid, "Resumen:");
    logRawLine(log, rid, "  - Aprobadas: " + std::to_string(passCount) + "/" + std::to_string(rep.size()));
    logRawLine(log, rid, "  - Umbral mínimo: " + std::to_string(QC_MIN_PASS));
    logRawLine(log, rid, "  - Decisión:     " + std::string(passCount >= QC_MIN_PASS ? "✓ CONTINUAR" : "✗ RECHAZAR"));
    logBlank(log, rid);
}

void logResumenLBP(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                          const std::vector<std::vector<double>>& features,
                          int totalImagenes) {
    if (LOG_DETAIL < 2 || features.empty()) return;
    
    logSection(log, rid, LOG_DETAIL, "RESUMEN: EXTRACCION LBP (Local Binary Patterns)");
    
    const int dims = (int)features[0].size();
    const int vectoresGenerados = (int)features.size();
    
    logRawLine(log, rid, "Configuración del descriptor:");
    logRawLine(log, rid, "  - Algoritmo:    LBP uniforme (rotacionalmente invariante)");
    logRawLine(log, rid, "  - Bloques:      4x4 (16 regiones)");
    logRawLine(log, rid, "  - Bins:         59 patrones por bloque");
    logRawLine(log, rid, "  - Dimensión:    " + std::to_string(dims) + " características (16 × 59)");
    logRawLine(log, rid, "  - Máscara ROI:  Aplicada (solo píxeles de oreja)");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Representación:");
    logRawLine(log, rid, "  Cada bloque genera un histograma de 59 bins que captura");
    logRawLine(log, rid, "  patrones de textura local (bordes, esquinas, áreas uniformes).");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Vectores generados:");
    logRawLine(log, rid, "  - Imágenes base:  " + std::to_string(totalImagenes));
    logRawLine(log, rid, "  - Con aumentación: ~" + std::to_string(totalImagenes * 7) + " (fotométrica)");
    logRawLine(log, rid, "  - Total extraído: " + std::to_string(vectoresGenerados));
    logRawLine(log, rid, "");
    
    // Mostrar samples de 3 vectores
    logRawLine(log, rid, "Samples de vectores (primeros 10 valores):");
    int samplesToShow = std::min(3, vectoresGenerados);
    for (int i = 0; i < samplesToShow; ++i) {
        logRawLine(log, rid, "  [" + std::to_string(i) + "] " + vecSample10(features[i]));
    }
    logBlank(log, rid);
}

void logTablaHoldout(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                            const std::vector<std::vector<double>>& Xh,
                            const std::vector<int>& yh,
                            const ModeloSVM& modelo,
                            int maxPrint,
                            int& outCorrect) {
    if (LOG_DETAIL < 2) {
        // Solo calcular correct sin imprimir
        outCorrect = 0;
        for (size_t i = 0; i < Xh.size(); ++i) {
            double best, second;
            int cls;
            int pred = predictOVAScore(modelo, Xh[i], best, second, cls);
            if (pred == yh[i]) outCorrect++;
        }
        return;
    }
    
    logRawLine(log, rid, "Evaluando " + std::to_string((int)Xh.size()) + " casos de holdout...");
    logRawLine(log, rid, "");
    
    outCorrect = 0;
    int toPrint = std::min(maxPrint, (int)Xh.size());
    
    for (int i = 0; i < toPrint; ++i) {
        const auto& x = Xh[i];
        int y = yh[i];
        
        double bestScore, secondScore;
        int bestClass;
        int pred = predictOVAScore(modelo, x, bestScore, secondScore, bestClass);
        double margen = bestScore - secondScore;
        
        bool ok = (pred == y);
        if (ok) outCorrect++;
        
        std::ostringstream oss;
        oss << "  Caso " << std::setw(3) << (i+1) << ": "
            << "y=" << std::setw(4) << y << " "
            << "pred=" << std::setw(4) << pred << " "
            << "margen=" << std::fixed << std::setprecision(2) << std::setw(5) << margen << " "
            << (ok ? "✓" : "✗");
        logRawLine(log, rid, oss.str());
    }
    
    // Evaluar el resto sin imprimir
    for (size_t i = toPrint; i < Xh.size(); ++i) {
        double best, second;
        int cls;
        int pred = predictOVAScore(modelo, Xh[i], best, second, cls);
        if (pred == yh[i]) outCorrect++;
    }
    
    if (toPrint < (int)Xh.size()) {
        logRawLine(log, rid, "  ... (mostrando primeros " + std::to_string(toPrint) + " casos)");
    }
    
    logRawLine(log, rid, "");
    double acc = 100.0 * (double)outCorrect / (double)Xh.size();
    logRawLine(log, rid, "Resumen:");
    logRawLine(log, rid, "  - Correctos: " + std::to_string(outCorrect) + "/" + std::to_string((int)Xh.size()));
    logRawLine(log, rid, "  - Accuracy:  " + std::to_string(acc) + "%");
    logBlank(log, rid);
}

void logResumenDuplicado(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                                int M, int votosConfiables, int claseMasVotada, int maxVotos,
                                double consistencia, double fracConfiables,
                                int minConfiables, int votosNecesarios,
                                double UMBRAL_MARGEN, double UMBRAL_CONSISTENCIA) {
    if (LOG_DETAIL < 2) return;
    
    logSection(log, rid, LOG_DETAIL, "ANALISIS ANTI-DUPLICADO BIOMETRICO");
    
    logRawLine(log, rid, "Método:");
    logRawLine(log, rid, "  Votación por margen de confianza sobre " + std::to_string(M) + " muestras");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Configuración:");
    logRawLine(log, rid, "  - Umbral margen:        >= " + std::to_string(UMBRAL_MARGEN));
    logRawLine(log, rid, "  - Umbral consistencia:  >= " + std::to_string(UMBRAL_CONSISTENCIA) + " (" + std::to_string((int)(UMBRAL_CONSISTENCIA*100)) + "%)");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Resultados:");
    logRawLine(log, rid, "  - Votos confiables:  " + std::to_string(votosConfiables) + "/" + std::to_string(M) + " (" + std::to_string((int)(fracConfiables*100)) + "%)");
    logRawLine(log, rid, "  - Clase más votada:  " + std::to_string(claseMasVotada));
    logRawLine(log, rid, "  - Votos recibidos:   " + std::to_string(maxVotos));
    logRawLine(log, rid, "  - Consistencia:      " + std::to_string((int)(consistencia*100)) + "% (" + std::to_string(maxVotos) + "/" + std::to_string(M) + ")");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Umbrales mínimos:");
    logRawLine(log, rid, "  - Votos confiables:  " + std::to_string(minConfiables) + " (necesario)");
    logRawLine(log, rid, "  - Votos necesarios:  " + std::to_string(votosNecesarios) + " (para coincidir)");
    logRawLine(log, rid, "");
    
    bool esDuplicado = (claseMasVotada != -1 && maxVotos >= votosNecesarios && votosConfiables >= minConfiables);
    
    logRawLine(log, rid, "Decisión:");
    if (esDuplicado) {
        logRawLine(log, rid, "  ✗ DUPLICADO PROBABLE");
        logRawLine(log, rid, "    La biometría coincide con clase existente: " + std::to_string(claseMasVotada));
    } else {
        logRawLine(log, rid, "  ✓ NO ES DUPLICADO");
        if (votosConfiables < minConfiables) {
            logRawLine(log, rid, "    Razón: Insuficientes votos confiables");
        } else if (maxVotos < votosNecesarios) {
            logRawLine(log, rid, "    Razón: Consistencia insuficiente (" + std::to_string((int)(consistencia*100)) + "% < " + std::to_string((int)(UMBRAL_CONSISTENCIA*100)) + "%)");
        } else {
            logRawLine(log, rid, "    Razón: No hay clase dominante");
        }
    }
    logBlank(log, rid);
}

void logTechniqueComparison(std::ofstream& log, const std::string& rid, 
                                   const StatsComparison& cmp) {
    logTechTitle(log, rid, cmp.tecnica);
    
    if (cmp.w_in == cmp.w_out && cmp.h_in == cmp.h_out) {
        logRawLine(log, rid, "Dimensiones: " + std::to_string(cmp.w_in) + "x" + std::to_string(cmp.h_in) + " (sin cambio)");
    } else {
        logRawLine(log, rid, "Entrada:  " + std::to_string(cmp.w_in) + "x" + std::to_string(cmp.h_in));
        logRawLine(log, rid, "Salida:   " + std::to_string(cmp.w_out) + "x" + std::to_string(cmp.h_out));
    }
    
    if (!cmp.params.empty()) {
        logRawLine(log, rid, "Params:   " + cmp.params);
    }
    
    logRawLine(log, rid, "");
    logRawLine(log, rid, "┌─────────────┬──────────┬──────────┬──────────┐");
    logRawLine(log, rid, "│ Métrica     │ Entrada  │ Salida   │ Delta    │");
    logRawLine(log, rid, "├─────────────┼──────────┼──────────┼──────────┤");
    
    {
        double delta = cmp.mean_out - cmp.mean_in;
        std::ostringstream oss;
        oss << "│ mean        │ " 
            << std::setw(8) << std::fixed << std::setprecision(2) << cmp.mean_in << " │ "
            << std::setw(8) << std::fixed << std::setprecision(2) << cmp.mean_out << " │ "
            << std::setw(8) << std::showpos << std::fixed << std::setprecision(2) << delta << " │";
        logRawLine(log, rid, oss.str());
    }
    
    {
        double delta = cmp.std_out - cmp.std_in;
        std::ostringstream oss;
        oss << "│ std         │ " 
            << std::setw(8) << std::fixed << std::setprecision(2) << cmp.std_in << " │ "
            << std::setw(8) << std::fixed << std::setprecision(2) << cmp.std_out << " │ "
            << std::setw(8) << std::showpos << std::fixed << std::setprecision(2) << delta << " │";
        logRawLine(log, rid, oss.str());
    }
    
    {
        int delta = cmp.min_out - cmp.min_in;
        std::ostringstream oss;
        oss << "│ min         │ " 
            << std::setw(8) << cmp.min_in << " │ "
            << std::setw(8) << cmp.min_out << " │ "
            << std::setw(8) << std::showpos << delta << " │";
        logRawLine(log, rid, oss.str());
    }
    
    {
        int delta = cmp.max_out - cmp.max_in;
        std::ostringstream oss;
        oss << "│ max         │ " 
            << std::setw(8) << cmp.max_in << " │ "
            << std::setw(8) << cmp.max_out << " │ "
            << std::setw(8) << std::showpos << delta << " │";
        logRawLine(log, rid, oss.str());
    }
    
    {
        double delta = cmp.pct_dark_out - cmp.pct_dark_in;
        std::ostringstream oss;
        oss << "│ pct_dark    │ " 
            << std::setw(6) << std::fixed << std::setprecision(1) << cmp.pct_dark_in << "% │ "
            << std::setw(6) << std::fixed << std::setprecision(1) << cmp.pct_dark_out << "% │ "
            << std::setw(6) << std::showpos << std::fixed << std::setprecision(1) << delta << "% │";
        logRawLine(log, rid, oss.str());
    }
    
    {
        double delta = cmp.pct_bright_out - cmp.pct_bright_in;
        std::ostringstream oss;
        oss << "│ pct_bright  │ " 
            << std::setw(6) << std::fixed << std::setprecision(1) << cmp.pct_bright_in << "% │ "
            << std::setw(6) << std::fixed << std::setprecision(1) << cmp.pct_bright_out << "% │ "
            << std::setw(6) << std::showpos << std::fixed << std::setprecision(1) << delta << "% │";
        logRawLine(log, rid, oss.str());
    }
    
    logRawLine(log, rid, "└─────────────┴──────────┴──────────┴──────────┘");
    logRawLine(log, rid, "");
    
    logRawLine(log, rid, "Tiempo:   " + std::to_string(cmp.ms) + " ms");
    if (!cmp.efecto.empty()) {
        logRawLine(log, rid, "Efecto:   " + cmp.efecto);
    }
    
    logBlank(log, rid);
}
