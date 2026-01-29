#pragma once
#include <string>
#include "admin/admin_types.h"

// MISMAS firmas (sin static)
std::string getEnv(const char* k, const std::string& def);
int getEnvInt(const char* k, int def);
double getEnvDouble(const char* k, double def);

ArgsBio parseArgsBio(int argc, char** argv);

QcThresholds loadQcThresholds();
Ctx loadCtxFromEnvAndArgs(const ArgsBio& a);
