from pathlib import Path
import io
import re
import sys
import json
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
import joblib
import shap

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION — modifica qui tutti i parametri
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

# --- Percorsi input ---
CSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "extracted_audio_features" / "audio" / "features_real_audio_dataset.csv"
TRAIN_TSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "real_audio_dataset.tsv"

# --- Percorsi output ---
OUTPUT_DIR   = PROJECT_ROOT / "classifiers" / "random forest" / "violence" / "toxic_audio_detection" / "real audio"
MODEL_DIR    = OUTPUT_DIR / "model"
PLOTS_DIR    = OUTPUT_DIR / "plots"
METRICS_DIR  = OUTPUT_DIR / "metrics"
FEATURES_DIR = OUTPUT_DIR / "top_features"

# --- Colonne da escludere dalle feature ---
EXCLUDE_COLUMNS = ["label", "sample_id", "path", "text"]

# --- Split train / test ---
TEST_SIZE    = 0.15
RANDOM_STATE = 42

# --- K-Fold (cross-validation esterna) ---
N_FOLDS       = 5
SHUFFLE_FOLDS = True

# --- Random Forest (usati solo se GRID_SEARCH_ENABLED=False) ---
RF_N_ESTIMATORS      = 100
RF_MAX_DEPTH         = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF  = 1
RF_MAX_FEATURES      = "sqrt"
RF_CLASS_WEIGHT      = "balanced"

# --- Grid / Randomized Search ---
GRID_SEARCH_ENABLED    = True
GRID_SEARCH_RANDOMIZED = True          # True = RandomizedSearchCV, False = GridSearchCV
GRID_SEARCH_N_ITER     = 30            # combinazioni campionate (solo se RANDOMIZED=True)
GRID_SEARCH_SCORING    = "f1_macro"    # metrica principale: "f1_macro" | "roc_auc" | "accuracy"
GRID_SEARCH_CV_FOLDS   = 5            # fold interni alla grid search
GRID_SEARCH_N_JOBS     = 1            # 1 = sequenziale (necessario per catturare stdout)
GRID_PARAM_GRID = {
    "classifier__n_estimators":      [100, 200, 300],
    "classifier__max_depth":         [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf":  [1, 2, 4],
    "classifier__max_features":      ["sqrt", "log2"],
}

# --- Pre-processing ---
SCALE_FEATURES     = False  # RF è invariante alla scala, di default False
DROP_CONSTANT_COLS = True   # rimuovi colonne con varianza zero
DROP_NA_ROWS       = False  # se False: sostituisce inf/NaN con 0 (come script originale)

# --- Top features per sample (explainability — SHAP) ---
TOP_K               = 10    # feature top/bottom per campione nel JSON
SHAP_MAX_DISPLAY    = 20    # feature mostrate nel beeswarm plot globale
SHAP_N_JOBS         = -1    # parallelismo TreeExplainer (-1 = tutti i core)

# --- Plot ---
CONFUSION_MATRIX_CMAP = "Blues"
FIGURE_DPI  = 150
FIGURE_SIZE = (8, 6)

# --- Logging ---
LOG_LEVEL   = logging.INFO
LOG_TO_FILE = True
LOG_FILE    = OUTPUT_DIR / "run.log"

# ==============================================================================
# SETUP DIRECTORY E LOGGER
# ==============================================================================

for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, METRICS_DIR, FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

handlers: list = [logging.StreamHandler()]
if LOG_TO_FILE:
    handlers.append(logging.FileHandler(LOG_FILE, mode="w"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)


# ==============================================================================
# 1. CARICAMENTO E FILTRAGGIO
# ==============================================================================

def load_data(csv_path: Path, train_tsv_path: Path) -> pd.DataFrame:
    """Carica il CSV delle feature e filtra solo le righe presenti nel TSV di training."""
    log.info(f"Caricamento feature CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    log.info(f"Caricamento train TSV:   {train_tsv_path}")
    train_df  = pd.read_csv(train_tsv_path, sep="\t")
    train_ids = set(train_df["id"])

    df_train = df[df["sample_id"].isin(train_ids)].reset_index(drop=True)
    log.info(f"  Totale campioni nel CSV:           {len(df)}")
    log.info(f"  Campioni nel TSV di training:      {len(train_ids)}")
    log.info(f"  Campioni dopo il filtraggio:       {len(df_train)}")
    return df_train


# ==============================================================================
# 2. PRE-PROCESSING
# ==============================================================================

def preprocess(df: pd.DataFrame):
    """Pulizia, selezione feature, encoding label."""
    if DROP_NA_ROWS:
        before = len(df)
        df = df.dropna()
        log.info(f"Rimozione righe con NaN: {before} -> {len(df)}")
    else:
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        log.info("Inf/NaN sostituiti con 0")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    if DROP_CONSTANT_COLS:
        variances  = df[feature_cols].var()
        const_cols = variances[variances == 0].index.tolist()
        if const_cols:
            log.info(f"Rimozione {len(const_cols)} colonne costanti")
            feature_cols = [c for c in feature_cols if c not in const_cols]

    X          = df[feature_cols].values.astype(np.float32)
    y          = df["label"].values
    sample_ids = df["sample_id"].values

    log.info(f"Feature usate: {len(feature_cols)}")
    log.info(f"Shape X: {X.shape}")
    log.info(f"Distribuzione label — 1 (tossico): {y.sum()}, 0 (non tossico): {(y == 0).sum()}")
    return X, y, feature_cols, sample_ids


# ==============================================================================
# 3. SPLIT TRAIN / TEST
# ==============================================================================

def split_data(X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray):
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, sample_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info(f"Split -> Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, ids_test


# ==============================================================================
# 4. PIPELINE (Scaler opzionale + Random Forest)
# ==============================================================================

def build_pipeline() -> Pipeline:
    steps = []
    if SCALE_FEATURES:
        steps.append(("scaler", StandardScaler()))
    steps.append((
        "classifier",
        RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight=RF_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ))
    return Pipeline(steps)


# ==============================================================================
# 5. GRID / RANDOMIZED SEARCH (opzionale)
# ==============================================================================

def run_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    total_combinations = _count_combinations(GRID_PARAM_GRID)
    n_iter      = GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else total_combinations
    search_type = "RandomizedSearchCV" if GRID_SEARCH_RANDOMIZED else "GridSearchCV"
    total_fits  = n_iter * GRID_SEARCH_CV_FOLDS

    log.info(f"\n{'=' * 60}")
    log.info(f"{search_type} (scoring={GRID_SEARCH_SCORING}, cv={GRID_SEARCH_CV_FOLDS})")
    log.info(f"Combinazioni totali nello spazio: {total_combinations}")
    log.info(f"Combinazioni che verranno testate: {n_iter}")
    log.info(f"Fit totali da eseguire: {total_fits}")
    log.info(f"{'=' * 60}")

    base_pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=GRID_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------ #
    # Intercetta stdout di scikit-learn (verbose=3) e lo converte in log  #
    # ------------------------------------------------------------------ #
    fit_counter     = {"n": 0}
    original_stdout = sys.stdout

    class ProgressCapture(io.TextIOBase):
        def write(self, text):
            text = text.strip()
            if not text:
                return len(text) if text is not None else 0
            if re.search(r"\[CV\s*\d+/\d+\]\s*END", text):
                fit_counter["n"] += 1
                score_match = re.search(r"test=([0-9.]+)", text)
                score_str   = f"  score={float(score_match.group(1)):.4f}" if score_match else ""
                log.info(f"  [{fit_counter['n']:>3}/{total_fits}] fit completato{score_str}")
            return len(text)

        def flush(self):
            pass

    sys.stdout = ProgressCapture()

    try:
        if GRID_SEARCH_RANDOMIZED:
            searcher = RandomizedSearchCV(
                estimator=base_pipe,
                param_distributions=GRID_PARAM_GRID,
                n_iter=GRID_SEARCH_N_ITER,
                scoring=GRID_SEARCH_SCORING,
                cv=cv,
                n_jobs=GRID_SEARCH_N_JOBS,
                verbose=3,
                refit=True,
                random_state=RANDOM_STATE,
            )
        else:
            searcher = GridSearchCV(
                estimator=base_pipe,
                param_grid=GRID_PARAM_GRID,
                scoring=GRID_SEARCH_SCORING,
                cv=cv,
                n_jobs=GRID_SEARCH_N_JOBS,
                verbose=3,
                refit=True,
            )

        searcher.fit(X_train, y_train)
    finally:
        sys.stdout = original_stdout

    log.info(f"\n  Best score ({GRID_SEARCH_SCORING}): {searcher.best_score_:.4f}")
    log.info(f"  Best params: {searcher.best_params_}")

    gs_results      = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
    gs_results_path = METRICS_DIR / "grid_search_results.csv"
    gs_results.to_csv(gs_results_path, index=False)
    log.info(f"  Risultati grid search salvati: {gs_results_path}")

    return {
        "best_params":   searcher.best_params_,
        "best_score":    float(searcher.best_score_),
        "best_pipeline": searcher.best_estimator_,
    }


def _count_combinations(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


# ==============================================================================
# 6. K-FOLD CROSS-VALIDATION (su train set) con threshold ottimale sulla PR curve
# ==============================================================================

def run_kfold(X_train: np.ndarray, y_train: np.ndarray, best_pipeline=None) -> dict:
    log.info(f"\n{'=' * 60}")
    log.info(f"K-Fold Cross-Validation (k={N_FOLDS}) sul training set")
    if best_pipeline is not None:
        log.info("  Usando i best params trovati dalla Grid Search")
    log.info(f"{'=' * 60}")

    import sklearn
    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)
    fold_metrics  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        pipe = sklearn.clone(best_pipeline) if best_pipeline is not None else build_pipeline()
        pipe.fit(X_tr, y_tr)

        probs = pipe.predict_proba(X_vl)[:, 1]

        y_pred = pipe.predict(X_vl)

        acc = accuracy_score(y_vl, y_pred)
        auc = roc_auc_score(y_vl, probs)
        f1  = f1_score(y_vl, y_pred, average="macro")
        cm  = confusion_matrix(y_vl, y_pred)
        rep = classification_report(
            y_vl, y_pred,
            target_names=["non-tossico", "tossico"],
            output_dict=True,
        )

        fold_metrics.append({
            "fold":                  fold,
            "accuracy":              float(acc),
            "roc_auc":               float(auc),
            "f1_macro":              float(f1),
            "confusion_matrix":      cm.tolist(),
            "classification_report": rep,
        })
        log.info(f"  Fold {fold}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}, F1-macro={f1:.4f}")
        log.info(f"\n{classification_report(y_vl, y_pred, target_names=['non-tossico', 'tossico'])}")

    accs = [m["accuracy"] for m in fold_metrics]
    aucs = [m["roc_auc"]  for m in fold_metrics]
    f1s  = [m["f1_macro"] for m in fold_metrics]

    log.info(f"\n  Media Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    log.info(f"  Media ROC-AUC   : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    log.info(f"  Media F1-macro  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    return {
        "folds":            fold_metrics,
        "mean_accuracy":    float(np.mean(accs)),
        "std_accuracy":     float(np.std(accs)),
        "mean_roc_auc":     float(np.mean(aucs)),
        "std_roc_auc":      float(np.std(aucs)),
        "mean_f1_macro":    float(np.mean(f1s)),
        "std_f1_macro":     float(np.std(f1s)),
    }


# ==============================================================================
# 7. TOP FEATURES PER SAMPLE (EXPLAINABILITY — SHAP)
# ==============================================================================

def save_top_features(
    pipe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    sample_ids: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    top_k: int = TOP_K,
):
    """Calcola i valori SHAP sul test set tramite TreeExplainer e salva:
      - top_features/test_set_top_features.json  : shap value per ogni sample
      - top_features/shap_values.npy             : matrice SHAP grezza (n_samples, n_features)
      - plots/shap_beeswarm.png                  : beeswarm plot globale (top SHAP_MAX_DISPLAY feature)

    TreeExplainer (Lundberg et al., 2020) calcola i valori di Shapley in modo
    esatto per gli ensemble ad albero in O(TLD), dove T = n_estimators,
    L = n_foglie per albero, D = profondità massima. I valori SHAP per la
    classe positiva (tossico) sono in shap_values[:, :, 1].

    Schema JSON (per sample):
      "top10_toxic"    : feature con shap_value > 0 più alto  → spingono verso classe 1
      "top10_nontoxic" : feature con shap_value < 0 più basso → spingono verso classe 0
    """
    clf = pipe.named_steps["classifier"]

    log.info("  Calcolo SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(clf)
    # shap_values: shape (n_samples, n_features, n_classes) per RF multi-output
    # oppure (n_samples, n_features) se binaria — gestiamo entrambi
    shap_values_raw = explainer.shap_values(X_test)

    # Normalizza a (n_samples, n_features) per la classe positiva (1 = tossico)
    if isinstance(shap_values_raw, list):
        # shap < 0.40: lista [class0, class1], ciascuna (n_samples, n_features)
        sv = np.array(shap_values_raw[1])
    elif shap_values_raw.ndim == 3:
        # shap >= 0.40: (n_samples, n_features, n_classes)
        sv = shap_values_raw[:, :, 1]
    else:
        # fallback: già (n_samples, n_features)
        sv = shap_values_raw

    log.info(f"  SHAP values shape: {sv.shape}")

    # ── Salvataggio matrice grezza ────────────────────────────────────────────
    npy_path = FEATURES_DIR / "shap_values.npy"
    np.save(npy_path, sv)
    log.info(f"  SHAP matrix salvata: {npy_path}")

    # ── Importanza globale = mean |SHAP| per feature ─────────────────────────
    mean_abs_shap   = np.abs(sv).mean(axis=0)           # (n_features,)
    top_global_idx  = np.argsort(mean_abs_shap)[-SHAP_MAX_DISPLAY:][::-1]
    global_importance = [
        {
            "feature":         feature_cols[j],
            "mean_abs_shap":   float(mean_abs_shap[j]),
        }
        for j in top_global_idx
    ]

    # ── Per sample: top-K feature per shap positivo e negativo ───────────────
    def _build_sample_entry(i: int) -> dict:
        x_raw    = X_test[i]
        sv_i     = sv[i]                                 # (n_features,)
        pos_idx  = np.argsort(sv_i)[-top_k:][::-1]      # top-K più positivi
        neg_idx  = np.argsort(sv_i)[:top_k]             # top-K più negativi

        return {
            "top10_toxic": [
                {
                    "feature":    feature_cols[j],
                    "value":      float(x_raw[j]),
                    "shap_value": float(sv_i[j]),        # contributo verso classe 1
                }
                for j in pos_idx
            ],
            "top10_nontoxic": [
                {
                    "feature":    feature_cols[j],
                    "value":      float(x_raw[j]),
                    "shap_value": float(sv_i[j]),        # contributo verso classe 0
                }
                for j in neg_idx
            ],
        }

    all_samples = {}
    for i, sample_id in enumerate(sample_ids):
        entry = _build_sample_entry(i)
        all_samples[str(sample_id)] = {
            "true_label":      int(y_test[i]),
            "predicted_prob":  float(probs[i]),
            "predicted_label": int(preds[i]),
            "expected_value":  float(explainer.expected_value[1]
                                     if isinstance(explainer.expected_value, (list, np.ndarray))
                                     else explainer.expected_value),
            **entry,
        }

    output = {
        "global_feature_importance": global_importance,
        "samples":                   all_samples,
    }

    json_path = FEATURES_DIR / "test_set_top_features.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)
    log.info(f"  Top features (SHAP) salvate: {json_path}")

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    _save_shap_beeswarm(sv, X_test, feature_cols)


def _save_shap_beeswarm(
    sv: np.ndarray,
    X_test: np.ndarray,
    feature_cols: list,
):
    """Salva il beeswarm plot SHAP (top SHAP_MAX_DISPLAY feature per |mean SHAP|)."""
    explanation = shap.Explanation(
        values          = sv,
        base_values     = np.zeros(sv.shape[0]),   # non serve per il beeswarm
        data            = X_test,
        feature_names   = feature_cols,
    )

    plt.figure(figsize=(10, 0.5 * SHAP_MAX_DISPLAY + 2))
    shap.plots.beeswarm(
        explanation,
        max_display = SHAP_MAX_DISPLAY,
        show        = False,
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "shap_beeswarm.png"
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info(f"  SHAP beeswarm plot salvato: {out_path}")


# ==============================================================================
# 8. ADDESTRAMENTO FINALE E VALUTAZIONE SUL TEST SET
# ==============================================================================

def train_and_evaluate(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    feature_cols: list,
    sample_ids_test: np.ndarray,
    best_pipeline=None,
) -> tuple:
    log.info(f"\n{'=' * 60}")
    log.info("Addestramento modello finale su tutto il training set")
    log.info(f"{'=' * 60}")

    # Se la grid search ha già fatto il refit, best_pipeline è già fittato sull'intero X_train
    pipe = best_pipeline if best_pipeline is not None else build_pipeline()
    if best_pipeline is None:
        pipe.fit(X_train, y_train)

    eval_results = {}
    for split_name, X_s, y_s in [("test", X_test, y_test)]:
        y_prob = pipe.predict_proba(X_s)[:, 1]
        y_pred = pipe.predict(X_s)

        acc = accuracy_score(y_s, y_pred)
        auc = roc_auc_score(y_s, y_prob)
        f1  = f1_score(y_s, y_pred, average="macro")
        cm  = confusion_matrix(y_s, y_pred)
        rep = classification_report(
            y_s, y_pred,
            target_names=["non-tossico", "tossico"],
            output_dict=True,
        )
        eval_results[split_name] = {
            "accuracy":              float(acc),
            "roc_auc":               float(auc),
            "f1_macro":              float(f1),
            "confusion_matrix":      cm.tolist(),
            "classification_report": rep,
        }
        log.info(f"[TEST] Accuracy={acc:.4f}, ROC-AUC={auc:.4f}, "f"F1-macro={f1:.4f}")
        log.info(f"\n{classification_report(y_s, y_pred, target_names=['non-tossico', 'tossico'])}")
        _save_confusion_matrix(cm, split_name, f"Confusion Matrix — {split_name.capitalize()} Set")

    # Top features sul test set
    log.info(f"\n{'=' * 60}")
    log.info("Salvataggio top features sul test set")
    log.info(f"{'=' * 60}")
    save_top_features(
        pipe=pipe,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        sample_ids=sample_ids_test,
        probs=pipe.predict_proba(X_test)[:, 1],
        preds=pipe.predict(X_test),
    )

    return pipe, eval_results


# ==============================================================================
# 9. SALVATAGGIO CONFUSION MATRIX
# ==============================================================================

def _save_confusion_matrix(cm: np.ndarray, split_name: str, title: str):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CONFUSION_MATRIX_CMAP,
        xticklabels=["non-tossico", "tossico"],
        yticklabels=["non-tossico", "tossico"],
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    out_path = PLOTS_DIR / f"confusion_matrix_{split_name}.png"
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info(f"  Confusion matrix salvata: {out_path}")


# ==============================================================================
# 10. SALVATAGGIO MODELLO E METRICHE
# ==============================================================================

def save_results(
    pipe,
    kfold_results: dict,
    eval_results: dict,
    feature_cols: list,
    grid_search_results: dict | None = None,
):
    # Classifier via joblib (compatibile con script originale)
    joblib.dump(pipe.named_steps["classifier"], MODEL_DIR / "best_audio_classifier.pkl")
    log.info(f"  Classifier salvato: {MODEL_DIR / 'best_audio_classifier.pkl'}")

    # Pipeline completa via pickle (per uso diretto in inference)
    pipeline_path = MODEL_DIR / "pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipe, f)
    log.info(f"  Pipeline completa salvata: {pipeline_path}")

    # config.json (compatibile con script originale — usato in inference)
    best_params = grid_search_results["best_params"] if grid_search_results else {
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth":    RF_MAX_DEPTH,
    }
    config = {
        "best_f1_cv":     float(kfold_results["mean_f1_macro"]),
        "best_params":    best_params,
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    log.info(f"  Config salvata: {MODEL_DIR / 'config.json'}")

    # metrics.json completo
    metrics = {
        "config": {
            "n_folds":      N_FOLDS,
            "test_size":    TEST_SIZE,
            "n_features":   len(feature_cols),
            "rf_class_weight": RF_CLASS_WEIGHT,
        },
        "kfold":      kfold_results,
        "evaluation": eval_results,
        "grid_search": {
            "enabled":     GRID_SEARCH_ENABLED,
            "randomized":  GRID_SEARCH_RANDOMIZED,
            "n_iter":      GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else None,
            "best_params": grid_search_results["best_params"] if grid_search_results else None,
            "best_score":  grid_search_results["best_score"]  if grid_search_results else None,
        },
    }
    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metriche salvate: {metrics_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    log.info("=" * 60)
    log.info("Toxic Audio Detection — Random Forest Classifier")
    log.info("=" * 60)
    log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # 1. Caricamento e filtraggio
    df = load_data(CSV_PATH, TRAIN_TSV_PATH)

    # 2. Pre-processing
    X, y, feature_cols, sample_ids = preprocess(df)

    # 3. Split train / test
    X_train, X_test, y_train, y_test, ids_test = split_data(X, y, sample_ids)

    # 4. Grid / Randomized Search (opzionale)
    grid_search_results = None
    best_pipeline       = None
    if GRID_SEARCH_ENABLED:
        grid_search_results = run_grid_search(X_train, y_train)
        best_pipeline       = grid_search_results["best_pipeline"]

    # 5. K-Fold CV sul training set (con threshold ottimale sulla PR curve)
    kfold_results  = run_kfold(X_train, y_train, best_pipeline)

    # 6. Training finale + valutazione test set
    pipe, eval_results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        feature_cols=feature_cols,
        sample_ids_test=ids_test,
        best_pipeline=best_pipeline,
    )

    # 7. Salvataggio
    save_results(pipe, kfold_results, eval_results, feature_cols, grid_search_results)

    log.info(f"\n{'=' * 60}")
    log.info("Pipeline completata con successo.")
    log.info(f"Output salvati in: {OUTPUT_DIR}")
    log.info(f"{'=' * 60}")

    log.info(f"\n[OK] Best F1-macro CV:  {kfold_results['mean_f1_macro']:.4f}")
    log.info(f"[OK] Best params:       {grid_search_results['best_params'] if grid_search_results else 'default'}")


if __name__ == "__main__":
    main()