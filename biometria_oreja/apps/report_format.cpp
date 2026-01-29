#include "report_format.h"

std::string repLine2(char ch, int n){
    return std::string(n, ch) + "\n";
}

std::string repTitle(const std::string& title, char ch){
    std::string out;
    out += repLine2(ch);
    out += "  " + title + "\n";
    out += repLine2(ch);
    return out;
}

std::string repSection(const std::string& title){
    return "\n" + repLine2('-') + "  " + title + "\n" + repLine2('-');
}

std::string repSub(const std::string& title){
    return "\n" + std::string(">> ") + title + "\n";
}

std::string repOK(const std::string& msg){   return "  [OK] " + msg + "\n"; }
std::string repWARN(const std::string& msg){ return "  [WARN] " + msg + "\n"; }
std::string repFAIL(const std::string& msg){ return "  [FAIL] " + msg + "\n"; }

std::string repKeyVal(const std::string& k, const std::string& v){
    return "  - " + k + ": " + v + "\n";
}

std::string repEndFail(const std::string& reason){
    return repSection("FINALIZACION") + repFAIL(reason);
}
