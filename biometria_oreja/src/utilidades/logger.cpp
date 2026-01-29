#include "utilidades/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <ctime>
#include <mutex>
#include <fstream>
#include <chrono>

static std::mutex g_log_mutex;
static LogLevel g_level = LOG_INFO;
static std::ofstream g_file;
static bool g_file_enabled = false;

static std::string nowTs() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static const char* levelStr(LogLevel lv) {
    switch (lv) {
        case LOG_DEBUG: return "DEBUG";
        case LOG_INFO:  return "INFO";
        case LOG_WARN:  return "WARN";
        case LOG_ERROR: return "ERROR";
        default:        return "UNK";
    }
}

void setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_level = level;
}

void setLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (g_file.is_open()) g_file.close();
    g_file.open(path, std::ios::app);
    g_file_enabled = g_file.is_open();
}

std::string makeRequestId() {
    // corto pero suficiente para defensa
    static thread_local std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dist(0, 15);

    std::ostringstream oss;
    oss << std::hex;
    for (int i = 0; i < 12; ++i) oss << dist(rng);
    return oss.str();
}

void logMessage(LogLevel level,
                const std::string& tag,
                const std::string& rid,
                const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_log_mutex);

    if (level < g_level) return;

    // Prefijo SOLO para la primera linea
    std::ostringstream prefix;
    prefix << "[" << levelStr(level) << "]"
        << " [" << tag << "] ";

    const std::string pfx = prefix.str();

    // Si msg tiene saltos de linea, las siguientes lineas NO repiten prefijo:
    // se imprimen alineadas con espacios del mismo ancho que el prefijo.
    std::istringstream iss(msg);
    std::string linePart;
    bool first = true;

    std::ostringstream out;

    while (std::getline(iss, linePart)) {
        if (first) {
            out << pfx << linePart << "\n";
            first = false;
        } else {
            out << std::string(pfx.size(), ' ') << linePart << "\n";
        }
    }

    // Caso especial: msg vacio (para que igual salga algo)
    if (first) {
        out << pfx << "\n";
    }

    // stdout/stderr (docker logs)
    std::cerr << out.str();
    std::cerr.flush();

    // archivo opcional
    if (g_file_enabled) {
        g_file << out.str();
        g_file.flush();
    }
}


LogScope::LogScope(const std::string& tag,
                   const std::string& rid,
                   const std::string& name)
    : tag_(tag), rid_(rid), name_(name), t0_(std::chrono::steady_clock::now()) {
    logMessage(LOG_INFO, tag_, rid_, "BEGIN " + name_);
}

LogScope::~LogScope() {
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0_).count();
    logMessage(LOG_INFO, tag_, rid_, "END " + name_ + " duration_ms=" + std::to_string(ms));
}
