#pragma once
#include <fstream>
#include <vector>

#include "admin/admin_types.h"
#include "admin/admin_log_helpers.h"
#include "svm/svm_entrenamiento.h"
#include "svm/svm_prediccion.h"

void startupLogs(std::ofstream& log, const Ctx& ctx);

void logTablaQC(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                       const std::vector<ImageReport>& rep,
                       const QcThresholds& QC, int QC_MIN_PASS);

void logBloquePorImagen(std::ofstream& log, const Ctx& ctx,
                               int idxPlus1, int total,
                               const std::string& ruta,
                               const ImageReport& r);

void logResumenLBP(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                          const std::vector<std::vector<double>>& features,
                          int totalImagenes);

void logTablaHoldout(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                            const std::vector<std::vector<double>>& Xh,
                            const std::vector<int>& yh,
                            const ModeloSVM& modelo,
                            int maxPrint,
                            int& outCorrect);

void logResumenDuplicado(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
                                int M, int votosConfiables, int claseMasVotada, int maxVotos,
                                double consistencia, double fracConfiables,
                                int minConfiables, int votosNecesarios,
                                double UMBRAL_MARGEN, double UMBRAL_CONSISTENCIA);

void logTechniqueComparison(std::ofstream& log, const std::string& rid, 
                                   const StatsComparison& cmp);
