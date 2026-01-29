#include "utilidades/zscore_params.h"
#include <fstream>
#include <sstream>
#include <cmath>

static bool parseLineDoubles(const std::string& line, std::vector<double>& out, char sep) {
    out.clear();
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        if (tok.empty()) continue;
        try { out.push_back(std::stod(tok)); }
        catch (...) { return false; }
    }
    return !out.empty();
}

bool guardarZScoreParams(const std::string& path, const ZScoreParams& p, char sep) {
    if (p.mean.empty() || p.stdev.empty() || p.mean.size() != p.stdev.size()) return false;
    std::ofstream f(path, std::ios::out);
    if (!f.is_open()) return false;

    // primera línea: dims
    f << p.mean.size() << "\n";

    // mean
    for (size_t i = 0; i < p.mean.size(); ++i) {
        if (i) f << sep;
        f << p.mean[i];
    }
    f << "\n";

    // stdev
    for (size_t i = 0; i < p.stdev.size(); ++i) {
        if (i) f << sep;
        f << p.stdev[i];
    }
    f << "\n";

    return true;
}

bool cargarZScoreParams(const std::string& path, ZScoreParams& p, char sep) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string lineDims, lineMean, lineStd;
    if (!std::getline(f, lineDims)) return false;
    if (!std::getline(f, lineMean)) return false;
    if (!std::getline(f, lineStd)) return false;

    size_t dims = 0;
    try { dims = (size_t)std::stoull(lineDims); }
    catch (...) { return false; }

    std::vector<double> mean, stdev;
    if (!parseLineDoubles(lineMean, mean, sep)) return false;
    if (!parseLineDoubles(lineStd, stdev, sep)) return false;

    if (mean.size() != dims || stdev.size() != dims) return false;
    for (auto& s : stdev) {
        if (std::fabs(s) < 1e-12) s = 1.0; // guard
    }

    p.mean = std::move(mean);
    p.stdev = std::move(stdev);
    return true;
}

bool aplicarZScore(std::vector<double>& x, const ZScoreParams& p) {
    if (x.size() != p.mean.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (x[i] - p.mean[i]) / p.stdev[i];
    }
    return true;
}

bool aplicarZScoreBatch(std::vector<std::vector<double>>& X, const ZScoreParams& p) {
    for (auto& v : X) {
        if (!aplicarZScore(v, p)) return false;
    }
    return true;
}
