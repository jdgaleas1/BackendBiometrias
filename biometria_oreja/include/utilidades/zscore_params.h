#pragma once
#include <vector>
#include <string>

struct ZScoreParams {
    std::vector<double> mean;
    std::vector<double> stdev;
};

bool guardarZScoreParams(const std::string& path, const ZScoreParams& p, char sep=';');
bool cargarZScoreParams(const std::string& path, ZScoreParams& p, char sep=';');
bool aplicarZScore(std::vector<double>& x, const ZScoreParams& p);
bool aplicarZScoreBatch(std::vector<std::vector<double>>& X, const ZScoreParams& p);
