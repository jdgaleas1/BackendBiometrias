#pragma once
#include <string>
#include <cstdint>
#include <chrono>

std::string nowTs();
std::string tsCompact();

using TimePoint = std::chrono::steady_clock::time_point;

TimePoint tick();
long long msSince(TimePoint t0);
