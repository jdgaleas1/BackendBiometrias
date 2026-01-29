#pragma once
#include <memory> 
#include <cstdint>
#include <string>
#include <vector>

struct QcThresholds {
    double mean_min = 49.314;     // P5  de qc_mean
    double mean_max = 90.156;     // P95 de qc_mean
    double std_min  = 23.294;     // P5  de qc_std
    int    min_min  = 10;         // se mantiene (no fue el limitante en tu base)
    int    max_max  = 245;        // se mantiene (no fue el limitante en tu base)
    double pct_dark_max   = 28.481; // P95 de qc_pct_dark
    double pct_bright_max = 0.001;  // P95 de qc_pct_bright (ojo: muy bajo, es correcto por tu base)
    int dark_thr = 10;
    int bright_thr = 245;
};

struct Ctx {
    std::string rid;
    std::string workDir;
    bool AUDIT_MODE = false;
    bool debug = false;

    int LOG_DETAIL = 2;
    int EVAL_PRINT_N = 40;

    double PERF_DROP_THRESHOLD = 2.0;

    QcThresholds QC;
    int QC_MIN_PASS = 6;
    int QC_ENFORCE = 0;

    std::string modelDir = "out";

    std::string rutaCSV;
    std::string rutaModeloPCA;
    std::string rutaModeloLDA;
    std::string rutaModeloSVM;

    std::string rutaTemplates;

    std::string rutaZScore;

    std::string holdoutCsv;
    std::string holdoutMetaJson;
    std::string baselineJson;
    std::string dirVersiones;

    double DUP_UMBRAL_MARGEN = 0.489;
    double DUP_UMBRAL_CONSISTENCIA = 0.70;
    double DUP_UMBRAL_VOTOS_CONFIABLES = 0.50;

    int POS_MAX = 250;
    int NEG_MAX = 800;
    double TRAIN_LR = 0.05;
    int TRAIN_EPOCHS = 400;
    double TRAIN_C = 1e-4;
    double TRAIN_TOL = 1e-4;
    int TRAIN_NEWNEG_MAX = 120;
    double TRAIN_LR2 = 0.01;
    double TRAIN_C2  = 1e-4;
};

struct GrayStats {
    double mean = 0.0;
    double stddev = 0.0;
    int minv = 255;
    int maxv = 0;
    double pct_dark = 0.0;
    double pct_bright = 0.0;
};

struct Imagen128 {
    std::unique_ptr<uint8_t[]> img128;
    std::unique_ptr<uint8_t[]> mask128;
    int w = 128;
    int h = 128;
};

struct StatsComparison {
    std::string tecnica;
    std::string params;
    
    int w_in = 0, h_in = 0;
    double mean_in = 0.0;
    double std_in = 0.0;
    int min_in = 0;
    int max_in = 0;
    double pct_dark_in = 0.0;
    double pct_bright_in = 0.0;
    
    int w_out = 0, h_out = 0;
    double mean_out = 0.0;
    double std_out = 0.0;
    int min_out = 0;
    int max_out = 0;
    double pct_dark_out = 0.0;
    double pct_bright_out = 0.0;
    
    long long ms = 0;
    std::string efecto;
};

struct ImageReport {
    std::string name;
    bool loadOk = false;
    bool qcOk = false;
    std::string qcReason;
    bool preprocOk = false;
    int augCount = 0;
    int featCount = 0;
    int dims = 0;
    long long ms_load = 0;
    long long ms_preproc = 0;
    long long ms_feats = 0;
    std::string err;
    
    // Stats para tabla QC (evaluados sobre ROI final)
    double mean = 0.0;
    double std = 0.0;
    int minv = 0;           // ← ESTE FALTABA
    int maxv = 0;           // ← ESTE FALTABA
    double pct_dark = 0.0;  // ← ESTE FALTABA
    double pct_bright = 0.0; // ← ESTE FALTABA
};

struct HoldoutMeta {
    int seed = 42;
    int total = 0;
    int test_size = 0;
    int dims = 0;
};

struct ArgsBio {
    std::string rid = "no-rid";
    bool debug = false;
};
