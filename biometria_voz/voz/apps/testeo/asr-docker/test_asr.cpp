#include "../../../core/asr/whisper_asr.h"
#include <iostream>

int main() {
    if (transcribeAndCompare("audio.flac", "El sol brilla en las montañas")) {
        std::cout << " Voz válida: coincidencia exacta.\n";
    }
    else {
        std::cout << " Voz inválida: no coincide con la frase esperada.\n";
    }

}
