#pragma once
#include <string>

std::string repLine2(char ch='=', int n=60);
std::string repTitle(const std::string& title, char ch='=');
std::string repSection(const std::string& title);
std::string repSub(const std::string& title);

std::string repOK(const std::string& msg);
std::string repWARN(const std::string& msg);
std::string repFAIL(const std::string& msg);

std::string repKeyVal(const std::string& k, const std::string& v);
std::string repEndFail(const std::string& reason);