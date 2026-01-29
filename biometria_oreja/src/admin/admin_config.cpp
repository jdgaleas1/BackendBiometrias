#include "admin/admin_config.h"

#include <cstdlib>
#include <string>

// Igual que en tu .cpp original
std::string getEnv(const char* k, const std::string& def) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : def;
}

int getEnvInt(const char* k, int def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

double getEnvDouble(const char* k, double def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try { return std::stod(v); } catch (...) { return def; }
}

ArgsBio parseArgsBio(int argc, char** argv) {
    ArgsBio a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--rid" && i + 1 < argc) a.rid = argv[++i];
        else if (s == "--debug") a.debug = true;
    }
    return a;
}

QcThresholds loadQcThresholds() {
    QcThresholds t;
    t.mean_min = getEnvDouble("QC_MEAN_MIN", t.mean_min);
    t.mean_max = getEnvDouble("QC_MEAN_MAX", t.mean_max);
    t.std_min  = getEnvDouble("QC_STD_MIN",  t.std_min);
    t.min_min  = getEnvInt   ("QC_MIN_MIN",  t.min_min);
    t.max_max  = getEnvInt   ("QC_MAX_MAX",  t.max_max);

    t.pct_dark_max   = getEnvDouble("QC_PCT_DARK_MAX",   t.pct_dark_max);
    t.pct_bright_max = getEnvDouble("QC_PCT_BRIGHT_MAX", t.pct_bright_max);

    t.dark_thr   = getEnvInt("QC_DARK_THR",   t.dark_thr);
    t.bright_thr = getEnvInt("QC_BRIGHT_THR", t.bright_thr);
    return t;
}

Ctx loadCtxFromEnvAndArgs(const ArgsBio& a) {
    Ctx ctx;
    ctx.rid = a.rid;
    ctx.debug = a.debug;

    ctx.AUDIT_MODE = (getEnv("AUDIT_MODE","0") == "1");
    ctx.workDir    = getEnv("WORK_DIR","nuevo_usuario");

    ctx.rutaZScore = ctx.modelDir + "/zscore_params.dat";

    ctx.LOG_DETAIL          = getEnvInt("LOG_DETAIL", 2);
    ctx.EVAL_PRINT_N        = getEnvInt("EVAL_PRINT_N", 40);
    ctx.PERF_DROP_THRESHOLD = getEnvDouble("PERF_DROP_THRESHOLD", 2.0);

    ctx.QC = loadQcThresholds();
    ctx.QC_MIN_PASS = getEnvInt("QC_MIN_PASS", 6);
    ctx.QC_ENFORCE  = getEnvInt("QC_ENFORCE", 0);

    ctx.modelDir = getEnv("MODEL_DIR","out");

    ctx.rutaCSV       = ctx.modelDir + "/caracteristicas_lda_train.csv";
    ctx.rutaModeloPCA = ctx.modelDir + "/modelo_pca.dat";
    ctx.rutaModeloLDA = ctx.modelDir + "/modelo_lda.dat";
    ctx.rutaModeloSVM = ctx.modelDir + "/modelo_svm.svm";
    ctx.rutaTemplates = ctx.modelDir + "/templates_k1.csv";

    ctx.holdoutCsv      = ctx.modelDir + "/holdout_test.csv";
    ctx.holdoutMetaJson = ctx.modelDir + "/holdout_meta.json";
    ctx.baselineJson    = ctx.modelDir + "/holdout_baseline.json";
    ctx.dirVersiones    = ctx.modelDir + "/versiones";

    // Si en tu Ctx ya tienes defaults, solo lee ENV aquí
    ctx.DUP_UMBRAL_MARGEN           = getEnvDouble("DUP_UMBRAL_MARGEN", ctx.DUP_UMBRAL_MARGEN);
    ctx.DUP_UMBRAL_CONSISTENCIA     = getEnvDouble("DUP_UMBRAL_CONSISTENCIA", ctx.DUP_UMBRAL_CONSISTENCIA);
    ctx.DUP_UMBRAL_VOTOS_CONFIABLES = getEnvDouble("DUP_UMBRAL_VOTOS_CONFIABLES", ctx.DUP_UMBRAL_VOTOS_CONFIABLES);

    // OJO: usa los mismos nombres que en tu main .cpp
    ctx.POS_MAX          = getEnvInt("TRAIN_POS_MAX", ctx.POS_MAX);
    ctx.NEG_MAX          = getEnvInt("TRAIN_NEG_MAX", ctx.NEG_MAX);
    ctx.TRAIN_LR         = getEnvDouble("TRAIN_LR", ctx.TRAIN_LR);
    ctx.TRAIN_EPOCHS     = getEnvInt("TRAIN_EPOCHS", ctx.TRAIN_EPOCHS);
    ctx.TRAIN_C          = getEnvDouble("TRAIN_C", ctx.TRAIN_C);
    ctx.TRAIN_TOL        = getEnvDouble("TRAIN_TOL", ctx.TRAIN_TOL);
    ctx.TRAIN_NEWNEG_MAX = getEnvInt("TRAIN_NEWNEG_MAX", ctx.TRAIN_NEWNEG_MAX);
    ctx.TRAIN_LR2        = getEnvDouble("TRAIN_LR2", ctx.TRAIN_LR2);
    ctx.TRAIN_C2         = getEnvDouble("TRAIN_C2", ctx.TRAIN_C2);

    return ctx;
}
