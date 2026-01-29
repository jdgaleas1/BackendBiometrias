#include "admin/admin_log_helpers.h"

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#else
  #error "Se requiere <filesystem> (C++17+)."
#endif

#include <iostream>
#include <string>

std::ofstream crearLogStream(const std::string& workDir) {
    try { fs::create_directories(workDir); } catch (...) {}
    return std::ofstream(workDir + "/agregar_usuario_biometria.log", std::ios::app);
}

std::string mkLine(const std::string& rid, const std::string& msg) {
    return std::string("[BIO] [rid=") + rid + "] " + msg;
}

void logRaw(std::ofstream& log, const std::string& text) {
#pragma omp critical(LOG_WRITE_BIO)
    {
        if (log.is_open()) {
            log << text;
            log.flush();
        }
        std::cerr << text;
        std::cerr.flush();
    }
}

void logRawLine(std::ofstream& log, const std::string& line) {
    logRaw(log, line + "\n");
}

// === Overloads de compatibilidad ===
void logRaw(std::ofstream& log, const std::string& rid, const std::string& textBlock) {
    (void)rid;                 // rid ya viene formateado afuera en muchos casos
    logRaw(log, textBlock);
}

void logRawLine(std::ofstream& log, const std::string& rid, const std::string& msg) {
    logRawLine(log, mkLine(rid, msg));
}

void logBlank(std::ofstream& log, const std::string& rid) {
    (void)rid;
    logRawLine(log, "");
}

void logMensaje(std::ofstream& log, const std::string& rid, const std::string& msg) {
    logRawLine(log, mkLine(rid, msg));
}

void logBlank(std::ofstream& log) {
    logRawLine(log, "");
}

void logTechTitle(std::ofstream& log, const std::string& rid, const std::string& title) {
    logMensaje(log, rid, "---- " + title + " ----");
}

void log7B(std::ofstream& log, const std::string& rid, const std::string& msg) {
    logMensaje(log, rid, std::string("[7B] ") + msg);
}

void logSection(std::ofstream& log, const std::string& rid, int LOG_DETAIL, const std::string& title) {
    if (LOG_DETAIL <= 0) return;
    logMensaje(log, rid, "============================================================");
    logMensaje(log, rid, title);
    logMensaje(log, rid, "============================================================");
}

void logPrettyTitle(std::ofstream& log, const Ctx& ctx, const std::string& title) {
    logSection(log, ctx.rid, ctx.LOG_DETAIL, title);
}

void logBox(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
            const std::string& title,
            const std::vector<std::string>& lines) {
    if (LOG_DETAIL <= 0) return;
    logMensaje(log, rid, "============================================================");
    logMensaje(log, rid, title);
    for (const auto& l : lines) logMensaje(log, rid, l);
    logMensaje(log, rid, "============================================================");
}

void logPhase(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
              int phaseNum, const std::string& phaseName,
              const std::string& objective,
              const std::vector<std::string>& lines) {
    if (LOG_DETAIL <= 0) return;
    logMensaje(log, rid, "------------------------------------------------------------");
    logMensaje(log, rid, "[FASE " + std::to_string(phaseNum) + "] " + phaseName);
    if (!objective.empty())
        logMensaje(log, rid, "Objetivo: " + objective);
    for (const auto& l : lines)
        logMensaje(log, rid, l);
    logMensaje(log, rid, "------------------------------------------------------------");
}

void logStep(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
             const std::string& stepId, const std::string& title) {
    if (LOG_DETAIL <= 0) return;
    logMensaje(log, rid, stepId + " " + title);
}

void logKV(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
           const std::string& k, const std::string& v, int indent) {
    if (LOG_DETAIL <= 0) return;
    std::string pad(indent, ' ');
    logMensaje(log, rid, pad + "- " + k + ": " + v);
}

void logOK(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
           const std::string& msg, int indent) {
    if (LOG_DETAIL <= 0) return;
    std::string pad(indent, ' ');
    logMensaje(log, rid, pad + "✓ " + msg);
}

void logWARN(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
             const std::string& msg, int indent) {
    if (LOG_DETAIL <= 0) return;
    std::string pad(indent, ' ');
    logMensaje(log, rid, pad + "⚠ " + msg);
}

void logERR(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
            const std::string& msg, int indent) {
    (void)LOG_DETAIL;
    std::string pad(indent, ' ');
    logMensaje(log, rid, pad + "✗ " + msg);
}

void logDet(std::ofstream& log, const std::string& rid, int LOG_DETAIL, int level, const std::string& msg) {
    if (LOG_DETAIL >= level)
        logMensaje(log, rid, msg);
}
