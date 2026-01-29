#pragma once
#include <string>

std::string getEnvStr(const char* key, const std::string& def_value = "");
std::string getEnvStr(const std::string& key, const std::string& def_value = "");

bool auditMode();
std::string tmpDir();
