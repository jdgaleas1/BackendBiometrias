#include "admin/admin_time.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

std::string nowTs() {
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

std::string tsCompact() {
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
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

TimePoint tick() {
    return std::chrono::steady_clock::now();
}

long long msSince(TimePoint t0) {
    auto t1 = std::chrono::steady_clock::now();
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}
