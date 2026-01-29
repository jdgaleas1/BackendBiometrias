#ifndef SIMILARIDAD_H
#define SIMILARIDAD_H

#include <string>

int distanciaLevenshtein(const std::string& s1, const std::string& s2);
double porcentajeSimilitud(const std::string& s1, const std::string& s2);

#endif
