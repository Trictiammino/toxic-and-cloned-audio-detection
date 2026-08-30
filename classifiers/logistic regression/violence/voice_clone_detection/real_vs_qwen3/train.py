from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
import io
import re
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import warnings
import json
import logging

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION — modifica qui tutti i parametri
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

# --- Percorsi input ---
CSV_CLONED_PATH = PROJECT_ROOT / "generate_cloned_audio" / "qwen3_tts" / "violence" / "extracted_audio_features" / "audio" / "features_cloned_audio_dataset.csv"  # label "Yes"
CSV_ORIGINAL_PATH = PROJECT_ROOT / "datasets" / "violence" / "extracted_audio_features" / "audio" / "features_real_audio_dataset.csv"  # label "No"

# --- Percorsi output ---
OUTPUT_DIR = PROJECT_ROOT / "classifiers" / "logistic regression" / "violence" / "voice_clone_detection" / "real_vs_qwen3"
MODEL_DIR = OUTPUT_DIR / "model"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

# --- Colonna label ---
LABEL_COLUMN = "cloned"
LABEL_CLONED = "Yes"
LABEL_ORIGINAL = "No"

# --- Split train / test (il validation è interno al K-Fold) ---
TEST_SIZE = 0.15  # frazione del totale per il test set
RANDOM_STATE = 42

# --- K-Fold ---
N_FOLDS = 5
SHUFFLE_FOLDS = True

# --- Logistic Regression ---
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"  # "lbfgs" | "saga" | "liblinear"
LR_C = 1.0  # inverso della regolarizzazione
LR_PENALTY = "l2"  # "l1" | "l2" | "elasticnet" | None
LR_CLASS_WEIGHT = "balanced"  # None | "balanced"

# --- Grid Search ---
GRID_SEARCH_ENABLED = True        # False per saltare la grid search e usare i parametri LR_* sopra
GRID_SEARCH_RANDOMIZED = True     # True = RandomizedSearchCV (più veloce), False = GridSearchCV (esaustiva)
GRID_SEARCH_N_ITER = 30           # numero di combinazioni campionate (usato solo se RANDOMIZED=True)
GRID_SEARCH_SCORING = "roc_auc"   # metrica usata per scegliere il best model: "roc_auc" | "accuracy" | "f1"
GRID_SEARCH_CV_FOLDS = 5          # fold interni alla grid search (separati dai N_FOLDS del K-Fold)
GRID_SEARCH_N_JOBS = -1           # parallelismo (-1 = tutti i core)
GRID_PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear", "saga"],
    "classifier__class_weight": [None, "balanced"],
    "classifier__max_iter": [500, 1000, 3000],
}

# --- Pre-processing ---
SCALE_FEATURES = True  # StandardScaler prima del modello
DROP_CONSTANT_COLS = True  # rimuovi colonne con varianza zero
DROP_NA_ROWS = True  # rimuovi righe con NaN
EXCLUDE_COLUMNS = ["sample_id", "path", "text", "label"]  # lista di colonne da escludere manualmente

# --- Plot ---
CONFUSION_MATRIX_CMAP = "Blues"
FIGURE_DPI = 150
FIGURE_SIZE = (8, 6)

# --- Logging ---
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True
LOG_FILE = OUTPUT_DIR / "run.log"

# ==============================================================================
# SETUP
# ==============================================================================

for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

handlers = [logging.StreamHandler()]
if LOG_TO_FILE:
    handlers.append(logging.FileHandler(LOG_FILE, mode="w"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)


# ==============================================================================
# 1. CARICAMENTO E MERGE DEI CSV
# ==============================================================================

def load_and_merge(cloned_path: Path, original_path: Path) -> pd.DataFrame:
    log.info(f"Caricamento CSV clonato:   {cloned_path}")
    df_cloned = pd.read_csv(cloned_path)
    df_cloned[LABEL_COLUMN] = LABEL_CLONED

    log.info(f"Caricamento CSV originale: {original_path}")
    df_original = pd.read_csv(original_path)
    df_original[LABEL_COLUMN] = LABEL_ORIGINAL

    log.info(f"  Righe cloned:   {len(df_cloned)}")
    log.info(f"  Righe original: {len(df_original)}")

    df = pd.concat([df_cloned, df_original], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log.info(f"  Totale dopo merge e shuffle: {len(df)} righe, {df.shape[1]} colonne")
    return df


# ==============================================================================
# 2. PRE-PROCESSING
# ==============================================================================

def preprocess(df: pd.DataFrame):
    if DROP_NA_ROWS:
        before = len(df)
        df = df.dropna()
        log.info(f"Rimozione righe con NaN: {before} -> {len(df)}")

    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]

    # Rimuovi colonne escluse manualmente
    if EXCLUDE_COLUMNS:
        excluded = [c for c in EXCLUDE_COLUMNS if c in feature_cols]
        if excluded:
            log.info(f"Colonne escluse manualmente: {excluded}")
            feature_cols = [c for c in feature_cols if c not in excluded]

    if DROP_CONSTANT_COLS:
        variances = df[feature_cols].var()
        const_cols = variances[variances == 0].index.tolist()
        if const_cols:
            log.info(f"Rimozione {len(const_cols)} colonne costanti")
            feature_cols = [c for c in feature_cols if c not in const_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = (df[LABEL_COLUMN] == LABEL_CLONED).astype(int).values  # 1=cloned, 0=original
    labels_str = np.where(y == 1, LABEL_CLONED, LABEL_ORIGINAL)

    log.info(f"Feature usate: {len(feature_cols)}")
    log.info(f"Distribuzione label — {LABEL_CLONED}: {y.sum()}, {LABEL_ORIGINAL}: {(y == 0).sum()}")
    return X, y, labels_str, feature_cols


# ==============================================================================
# 3. SPLIT TRAIN / TEST
# ==============================================================================

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info(f"Split -> Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 4. PIPELINE (Scaler + LogReg)
# ==============================================================================

def build_pipeline() -> Pipeline:
    steps = []
    if SCALE_FEATURES:
        steps.append(("scaler", StandardScaler()))
    steps.append((
        "classifier",
        LogisticRegression(
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            C=LR_C,
            penalty=LR_PENALTY,
            class_weight=LR_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),
    ))
    return Pipeline(steps)


# ==============================================================================
# 5. GRID SEARCH (opzionale)
# ==============================================================================

def run_grid_search(X_train, y_train) -> dict:
    total_combinations = _count_combinations(GRID_PARAM_GRID)
    n_iter = GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else total_combinations
    search_type = "RandomizedSearchCV" if GRID_SEARCH_RANDOMIZED else "GridSearchCV"
    total_fits = n_iter * GRID_SEARCH_CV_FOLDS

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
    fit_counter = {"n": 0}
    original_stdout = sys.stdout

    class ProgressCapture(io.TextIOBase):
        """Redirige stdout verso il logger, contando i fit completati."""
        def write(self, text):
            text = text.strip()
            if not text:
                return len(text) if text is not None else 0
            # sklearn con verbose≥2 emette righe tipo "[CV 1/3] END ..."
            if re.search(r"\[CV\s*\d+/\d+\]\s*END", text):
                fit_counter["n"] += 1
                # Estrai score dalla riga sklearn, es: "roc_auc: (test=0.9812)"
                score_match = re.search(r"test=([0-9.]+)", text)
                score_str = f"  score={float(score_match.group(1)):.4f}" if score_match else ""
                log.info(
                    f"  [{fit_counter['n']:>3}/{total_fits}] fit completato{score_str}"
                )
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
                n_jobs=1,          # 1 per permettere la cattura stdout
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
                n_jobs=1,          # 1 per permettere la cattura stdout
                verbose=3,
                refit=True,
            )

        searcher.fit(X_train, y_train)
    finally:
        sys.stdout = original_stdout  # ripristina sempre stdout

    log.info(f"\n  Best score ({GRID_SEARCH_SCORING}): {searcher.best_score_:.4f}")
    log.info(f"  Best params: {searcher.best_params_}")

    gs_results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
    gs_results_path = METRICS_DIR / "grid_search_results.csv"
    gs_results.to_csv(gs_results_path, index=False)
    log.info(f"  Risultati grid search salvati: {gs_results_path}")

    return {
        "best_params": searcher.best_params_,
        "best_score": float(searcher.best_score_),
        "best_pipeline": searcher.best_estimator_,
    }


def _count_combinations(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


# ==============================================================================
# 6. K-FOLD CROSS-VALIDATION (su train set)
# ==============================================================================

def run_kfold(X_train, y_train, best_pipeline=None) -> dict:
    log.info(f"\n{'=' * 60}")
    log.info(f"K-Fold Cross-Validation (k={N_FOLDS}) sul training set")
    if best_pipeline is not None:
        log.info("  Usando i best params trovati dalla Grid Search")
    log.info(f"{'=' * 60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=SHUFFLE_FOLDS, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        import sklearn
        if best_pipeline is not None:
            pipe = sklearn.clone(best_pipeline)
        else:
            pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_vl)
        y_prob = pipe.predict_proba(X_vl)[:, 1]

        acc = accuracy_score(y_vl, y_pred)
        auc = roc_auc_score(y_vl, y_prob)
        cm = confusion_matrix(y_vl, y_pred)
        rep = classification_report(
            y_vl, y_pred,
            target_names=[LABEL_ORIGINAL, LABEL_CLONED],
            output_dict=True,
        )

        fold_metrics.append({
            "fold": fold,
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": rep,
        })
        log.info(f"  Fold {fold}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
        log.info(f"\n{classification_report(y_vl, y_pred, target_names=[LABEL_ORIGINAL, LABEL_CLONED])}")

    accs = [m["accuracy"] for m in fold_metrics]
    aucs = [m["roc_auc"] for m in fold_metrics]
    log.info(f"\n  Media Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    log.info(f"  Media ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return {
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_roc_auc": float(np.mean(aucs)),
        "std_roc_auc": float(np.std(aucs)),
    }


# ==============================================================================
# 7. ADDESTRAMENTO FINALE E VALUTAZIONE
# ==============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, best_pipeline=None):
    log.info(f"\n{'=' * 60}")
    log.info("Addestramento modello finale su tutto il training set")
    log.info(f"{'=' * 60}")

    pipe = best_pipeline if best_pipeline is not None else build_pipeline()
    if best_pipeline is None:
        pipe.fit(X_train, y_train)

    results = {}
    for split_name, X_s, y_s in [
        ("test", X_test, y_test),
    ]:
        y_pred = pipe.predict(X_s)
        y_prob = pipe.predict_proba(X_s)[:, 1]
        acc = accuracy_score(y_s, y_pred)
        auc = roc_auc_score(y_s, y_prob)
        cm = confusion_matrix(y_s, y_pred)
        rep = classification_report(
            y_s, y_pred,
            target_names=[LABEL_ORIGINAL, LABEL_CLONED],
            output_dict=True,
        )
        results[split_name] = {
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": rep,
        }
        log.info(f"\n  [{split_name.upper()}] Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
        log.info(f"\n{classification_report(y_s, y_pred, target_names=[LABEL_ORIGINAL, LABEL_CLONED])}")

        _save_confusion_matrix(cm, split_name, f"Confusion Matrix — {split_name.capitalize()} Set")

    return pipe, results


# ==============================================================================
# 8. SALVATAGGIO MATRICE DI CONFUSIONE
# ==============================================================================

def _save_confusion_matrix(cm: np.ndarray, split_name: str, title: str):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CONFUSION_MATRIX_CMAP,
        xticklabels=[LABEL_ORIGINAL, LABEL_CLONED],
        yticklabels=[LABEL_ORIGINAL, LABEL_CLONED],
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
# 9. SALVATAGGIO METRICHE E MODELLO
# ==============================================================================

def save_results(pipe, kfold_results: dict, eval_results: dict, feature_cols: list, grid_search_results: dict = None):
    import pickle

    # Modello serializzato
    model_path = MODEL_DIR / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    log.info(f"\nModello salvato: {model_path}")

    # Metriche JSON
    metrics = {
        "config": {
            "n_folds": N_FOLDS,
            "test_size": TEST_SIZE,
            "lr_C": LR_C,
            "lr_penalty": LR_PENALTY,
            "lr_solver": LR_SOLVER,
            "lr_class_weight": LR_CLASS_WEIGHT,
            "n_features": len(feature_cols),
        },
        "kfold": kfold_results,
        "evaluation": eval_results,
        "grid_search": {
            "enabled": GRID_SEARCH_ENABLED,
            "randomized": GRID_SEARCH_RANDOMIZED,
            "n_iter": GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else None,
            "best_params": grid_search_results["best_params"] if grid_search_results else None,
            "best_score": grid_search_results["best_score"] if grid_search_results else None,
        },
    }
    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metriche salvate: {metrics_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    log.info("=" * 60)
    log.info("Audio Clone Detection — Logistic Regression Classifier")
    log.info("=" * 60)
    log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # 1. Carica e unisci i CSV
    df = load_and_merge(CSV_CLONED_PATH, CSV_ORIGINAL_PATH)

    # Salva il CSV merged per riferimento futuro
    merged_path = OUTPUT_DIR / "merged_dataset.csv"
    df.to_csv(merged_path, index=False)
    log.info(f"Dataset merged salvato: {merged_path}")

    # 2. Pre-processing
    X, y, _, feature_cols = preprocess(df)

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Grid Search (opzionale) — trova i best params sul training set
    grid_search_results = None
    best_pipeline = None
    if GRID_SEARCH_ENABLED:
        grid_search_results = run_grid_search(X_train, y_train)
        best_pipeline = grid_search_results["best_pipeline"]

    # 5. K-Fold CV sul training set (con i best params se grid search abilitata)
    kfold_results = run_kfold(X_train, y_train, best_pipeline)

    # 6. Training finale + evaluation sul test set
    pipe, eval_results = train_and_evaluate(X_train, X_test, y_train, y_test, best_pipeline)

    # 7. Salvataggio
    save_results(pipe, kfold_results, eval_results, feature_cols, grid_search_results)

    log.info(f"\n{'=' * 60}")
    log.info("Pipeline completata con successo.")
    log.info(f"Output salvati in: {OUTPUT_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()