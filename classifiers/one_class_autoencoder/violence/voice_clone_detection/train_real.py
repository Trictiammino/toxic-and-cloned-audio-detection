from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, GridSearchCV, train_test_split,
)
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils import resample
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
import pickle

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION -- modifica qui tutti i parametri
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# --- Percorso audio reali ---
CSV_ORIGINAL_PATH = (
    PROJECT_ROOT / "datasets" / "violence" / "extracted_audio_features"
    / "audio" / "features_real_audio_dataset.csv"
)

# --- Percorsi audio clonati ---
CSV_QWEN3_PATH = (PROJECT_ROOT / "generate_cloned_audio" / "qwen3_tts" / "violence" / "extracted_audio_features" / "audio" / "features_cloned_audio_dataset.csv")
CSV_CHATTERBOX_PATH = (PROJECT_ROOT / "generate_cloned_audio" / "chatterbox_tts" / "violence" / "extracted_audio_features" / "audio" / "features_cloned_audio_dataset.csv")

CLONED_SOURCES = {
    "qwen3":      CSV_QWEN3_PATH,
    "chatterbox": CSV_CHATTERBOX_PATH,
}

CLONED_SOURCE_COLORS = ["tomato", "darkorange", "mediumseagreen", "mediumpurple", "sienna"]

# --- Percorsi output ---
OUTPUT_DIR  = PROJECT_ROOT / "classifiers" / "one_class_autoencoder" / "violence" / "voice_clone_detection" / "train_real_results"
MODEL_DIR   = OUTPUT_DIR / "model"
PLOTS_DIR   = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

# --- Etichette ---
LABEL_NORMAL  = "Real"
LABEL_ANOMALY = "Cloned"

# --- Holdout test split ---
TEST_SIZE_REAL    = 0.10
TEST_SIZE_CLONED  = 0.20

# ------------------------------------------------------------------------
# FEATURE SELECTION / RIDUZIONE DIMENSIONALE
# ------------------------------------------------------------------------
USE_VARIANCE_THRESHOLD = True
VARIANCE_THRESHOLD     = 1e-4

USE_FEATURE_SELECTION  = True
N_FEATURES_SELECT      = 40

USE_PCA                = False
PCA_N_COMPONENTS       = 50

# --- One-Class Autoencoder (parametri default se GRID_SEARCH_ENABLED=False) ---
AE_HIDDEN_DIMS          = (128, 64)
AE_LATENT_DIM           = 8
AE_LEARNING_RATE        = 1e-3
AE_EPOCHS               = 300
AE_BATCH_SIZE           = 64
AE_DROPOUT_RATE         = 0.2
AE_ACTIVATION           = "relu"
AE_THRESHOLD_PERCENTILE = 90.0
AE_WEIGHT_DECAY         = 1e-4
AE_EARLY_STOPPING       = True
AE_PATIENCE             = 30
AE_USE_GPU              = True

# ------------------------------------------------------------------------
# ENSEMBLE (bagging) DI AUTOENCODER PER IL MODELLO FINALE
# ------------------------------------------------------------------------
AE_ENSEMBLE_SIZE      = 5
AE_ENSEMBLE_BOOTSTRAP = True

# --- K-Fold paired (diagnostica out-of-sample su train/val) ---
N_FOLDS       = 5
SHUFFLE_FOLDS = True
RANDOM_STATE  = 42

# ------------------------------------------------------------------------
# NOVITA' v3: CALIBRAZIONE ROBUSTA DELLA SOGLIA (bootstrap + vincolo FPR)
# ------------------------------------------------------------------------
# Nella v2 il metodo "youden" sceglieva la soglia scansionando OGNI valore
# di errore osservato nel calibration set e prendendo quello che massimizza
# J = TPR - FPR. Con un calibration set relativamente piccolo questo
# overfitta facilmente al rumore del singolo split: nei risultati osservati,
# il FPR sul test set (39.1%) era piu' del doppio del FPR medio visto in
# k-fold con soglia percentile (18.7%), a fronte di una ROC-AUC stabile
# (~0.77-0.79 sia in k-fold sia in grid search sia sul test) -- prova che il
# modello separa le classi in modo consistente, ma la SOGLIA scelta non
# generalizzava al di fuori del calibration set.
#
# Questa versione introduce, selezionabili via THRESHOLD_CALIBRATION_METHOD:
#
#   "percentile"       -> comportamento originale v1: percentile sui soli
#                          reali di calibrazione (non usa i clonati).
#
#   "youden"            -> come nella v2, ma i candidati sono una griglia di
#                          percentili invece di ogni valore osservato (gia'
#                          meno sensibile a singoli outlier del calibration
#                          set).
#
#   "youden_robust"     -> come "youden", ma la soglia finale e' la MEDIANA
#                          delle soglie ottime trovate su THRESHOLD_N_BOOTSTRAP
#                          resample bootstrap del calibration set -> riduce
#                          la varianza dovuta al singolo split.
#
#   "fpr_constrained"   -> (CONSIGLIATO per abbassare il FPR) invece di
#                          massimizzare J senza vincoli, sceglie la soglia
#                          con il TPR (detection rate) piu' alto TRA quelle
#                          che rispettano FPR <= THRESHOLD_TARGET_FPR sul
#                          calibration set. Stesso bootstrap di
#                          "youden_robust" per la stabilita'. Se nessun
#                          candidato rispetta il vincolo, ripiega sulla
#                          soglia con il FPR piu' basso disponibile nella
#                          griglia (mai peggio del vincolo per costruzione).
#
# Il default e' "fpr_constrained" con THRESHOLD_TARGET_FPR=0.10: limita
# esplicitamente i falsi allarmi su audio REALE, invece di lasciare che
# Youden li scambi liberamente con qualche punto di detection rate in piu'.
THRESHOLD_CALIBRATION_METHOD = "fpr_constrained"  # "percentile" | "youden" | "youden_robust" | "fpr_constrained"
CALIBRATION_SIZE             = 0.25               # aumentata da 0.15: soglia stimata su piu' campioni -> piu' stabile
THRESHOLD_TARGET_FPR         = 0.25               # usato solo con "fpr_constrained": FPR massimo tollerato sui reali
THRESHOLD_N_BOOTSTRAP        = 300                # ripetizioni bootstrap per "youden_robust" / "fpr_constrained"
THRESHOLD_N_CANDIDATES       = 300                # punti della griglia di percentili usata come soglie candidate

# --- Randomized / Grid Search ---
GRID_SEARCH_ENABLED    = True
GRID_SEARCH_RANDOMIZED = True
GRID_SEARCH_N_ITER     = 40
GRID_SEARCH_CV_FOLDS   = 5
GRID_SEARCH_N_JOBS     = -1

GRID_SEARCH_USE_AUC_SCORING = True
GRID_SEARCH_SCORING = "roc_auc_paired" if GRID_SEARCH_USE_AUC_SCORING else "balanced_autoencoder"

GRID_PARAM_GRID = {
    "classifier__hidden_dims":          [(128, 64), (256, 128), (256,256),],
    "classifier__latent_dim":           [2, 4, 8,],
    "classifier__learning_rate":        [5e-5, 1e-4, 3e-4, 5e-4],
    "classifier__epochs":               [200, 300, 400],
    "classifier__dropout_rate":         [0.0, 0.1, 0.2, 0.3],
    "classifier__weight_decay":         [1e-6, 1e-5, 1e-4],
}

# --- Pre-processing ---
SCALE_FEATURES     = True
SCALER_TYPE         = "robust"
DROP_CONSTANT_COLS = True
DROP_NA_ROWS       = True
EXCLUDE_COLUMNS    = ["sample_id", "path", "text", "label"]

# --- Plot ---
CONFUSION_MATRIX_CMAP = "Blues"
FIGURE_DPI  = 150
FIGURE_SIZE = (8, 6)

# --- Logging ---
LOG_LEVEL   = logging.INFO
LOG_TO_FILE = True
LOG_FILE    = OUTPUT_DIR / "run.log"


# ==============================================================================
# SETUP
# ==============================================================================

for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

try:
    _console_stream = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False, buffering=1)
except Exception:
    _console_stream = sys.stdout
handlers = [logging.StreamHandler(_console_stream)]
if LOG_TO_FILE:
    handlers.append(logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)


# ==============================================================================
# 1. CARICAMENTO DEI CSV
# ==============================================================================

def load_datasets(original_path: Path, cloned_sources: dict) -> tuple:
    log.info(f"Caricamento CSV audio reali: {original_path}")
    df_real = pd.read_csv(original_path)
    log.info(f"  Righe audio reali: {len(df_real)}")

    cloned_dfs = {}
    for name, path in cloned_sources.items():
        log.info(f"Caricamento CSV clonati [{name:>12}]: {path}")
        df = pd.read_csv(path)
        cloned_dfs[name] = df
        log.info(f"  Righe [{name}]: {len(df)}")

    return df_real, cloned_dfs


# ==============================================================================
# 2. PRE-PROCESSING + HOLDOUT TEST SPLIT + FEATURE SELECTION
# ==============================================================================

def _mutual_info_score_func(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


def preprocess_datasets(df_real: pd.DataFrame, cloned_dfs: dict) -> tuple:
    if DROP_NA_ROWS:
        before = len(df_real)
        df_real = df_real.dropna()
        log.info(f"Rimozione NaN (real): {before} -> {len(df_real)}")
        for name in list(cloned_dfs.keys()):
            before = len(cloned_dfs[name])
            cloned_dfs[name] = cloned_dfs[name].dropna()
            log.info(f"Rimozione NaN ({name:>12}): {before} -> {len(cloned_dfs[name])}")

    feature_cols = list(df_real.columns)

    if EXCLUDE_COLUMNS:
        excluded = [c for c in EXCLUDE_COLUMNS if c in feature_cols]
        if excluded:
            log.info(f"Colonne escluse manualmente: {excluded}")
            feature_cols = [c for c in feature_cols if c not in excluded]

    if DROP_CONSTANT_COLS:
        variances = df_real[feature_cols].var()
        const_cols = variances[variances == 0].index.tolist()
        if const_cols:
            log.info(f"Rimozione {len(const_cols)} colonne costanti (calcolate su real)")
            feature_cols = [c for c in feature_cols if c not in const_cols]

    for name, df in cloned_dfs.items():
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            log.warning(f"  {len(missing)} colonne mancanti nel CSV '{name}': {missing}")
            feature_cols = [c for c in feature_cols if c in df.columns]

    X_real = df_real[feature_cols].values.astype(np.float32)
    cloned_arrays = {
        name: df[feature_cols].values.astype(np.float32)
        for name, df in cloned_dfs.items()
    }

    X_real_tv, X_real_test = train_test_split(
        X_real, test_size=TEST_SIZE_REAL, random_state=RANDOM_STATE, shuffle=True
    )
    cloned_tv   = {}
    cloned_test = {}
    for name, X in cloned_arrays.items():
        X_tv, X_t = train_test_split(
            X, test_size=TEST_SIZE_CLONED, random_state=RANDOM_STATE, shuffle=True
        )
        cloned_tv[name]   = X_tv
        cloned_test[name] = X_t

    log.info(f"\nFeature disponibili (prima della feature selection): {len(feature_cols)}")
    log.info(f"Campioni reali:  train/val={len(X_real_tv)}  |  test={len(X_real_test)}")
    for name in cloned_tv:
        log.info(
            f"Campioni [{name:>12}]:  "
            f"train/val={len(cloned_tv[name])}  |  test={len(cloned_test[name])}"
        )

    if USE_VARIANCE_THRESHOLD:
        vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        vt.fit(X_real_tv)
        keep_mask = vt.get_support()
        n_removed = int(np.sum(~keep_mask))
        if n_removed > 0:
            log.info(f"VarianceThreshold: rimosse {n_removed} feature a bassa varianza "
                      f"(soglia={VARIANCE_THRESHOLD})")
            X_real_tv   = X_real_tv[:, keep_mask]
            X_real_test = X_real_test[:, keep_mask]
            for name in cloned_tv:
                cloned_tv[name]   = cloned_tv[name][:, keep_mask]
                cloned_test[name] = cloned_test[name][:, keep_mask]
            feature_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]

    feature_selector = None
    if USE_FEATURE_SELECTION:
        X_fs_train = np.vstack([X_real_tv] + list(cloned_tv.values()))
        y_fs_train = np.concatenate([
            np.zeros(len(X_real_tv), dtype=int),
            *[np.ones(len(v), dtype=int) for v in cloned_tv.values()],
        ])
        k = min(N_FEATURES_SELECT, X_fs_train.shape[1])
        feature_selector = SelectKBest(
            score_func=_mutual_info_score_func,
            k=k,
        )
        feature_selector.fit(X_fs_train, y_fs_train)
        support = feature_selector.get_support()
        selected_cols = [c for c, keep in zip(feature_cols, support) if keep]
        log.info(f"SelectKBest (mutual information): {X_fs_train.shape[1]} -> {k} feature")
        log.info(f"  Top 15 feature selezionate: {selected_cols[:15]}")

        X_real_tv   = feature_selector.transform(X_real_tv)
        X_real_test = feature_selector.transform(X_real_test)
        for name in cloned_tv:
            cloned_tv[name]   = feature_selector.transform(cloned_tv[name])
            cloned_test[name] = feature_selector.transform(cloned_test[name])
        feature_cols = selected_cols

    log.info(f"\nFeature finali usate dal modello: {len(feature_cols)}")

    return X_real_tv, X_real_test, cloned_tv, cloned_test, feature_cols, feature_selector


# ==============================================================================
# 3. RETE NEURALE AUTOENCODER
# ==============================================================================

class _AutoencoderNet(nn.Module):
    _ACTIVATIONS = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple,
        latent_dim: int,
        dropout_rate: float,
        activation: str,
    ):
        super().__init__()
        act_fn = self._ACTIVATIONS[activation]

        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), act_fn(), nn.Dropout(dropout_rate)])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), act_fn(), nn.Dropout(dropout_rate)])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ==============================================================================
# 3b. SELEZIONE DELLA SOGLIA (funzioni condivise, usate sia da
#     OneClassAutoencoder sia da OneClassAutoencoderEnsemble)
# ==============================================================================

def _best_threshold_on_sample(
    err_real: np.ndarray,
    err_anomaly: np.ndarray,
    candidates: np.ndarray,
    mode: str,
    target_fpr: float,
) -> tuple:
    """
    Cerca, tra i candidati, la soglia migliore secondo `mode`:
      "youden"          -> massimizza TPR - FPR (nessun vincolo)
      "fpr_constrained" -> massimizza TPR tra i candidati con FPR <= target_fpr;
                           se nessun candidato rispetta il vincolo, ripiega
                           sul candidato con il FPR piu' basso disponibile
                           nella griglia (soglia piu' conservativa possibile).
    """
    best_score, best_thr = -np.inf, None
    fallback_thr, fallback_fpr = float(candidates[-1]), np.inf

    for thr in candidates:
        fpr = float(np.mean(err_real > thr))
        tpr = float(np.mean(err_anomaly > thr))

        if fpr < fallback_fpr:
            fallback_fpr, fallback_thr = fpr, float(thr)

        if mode == "youden":
            score = tpr - fpr
        elif mode == "fpr_constrained":
            score = tpr if fpr <= target_fpr else -np.inf
        else:
            raise ValueError(f"mode sconosciuto: {mode}")

        if score > best_score:
            best_score, best_thr = score, float(thr)

    if best_thr is None or best_score == -np.inf:
        best_thr, best_score = fallback_thr, -np.inf

    return best_thr, best_score


def _select_threshold(
    err_real: np.ndarray,
    err_anomaly: np.ndarray = None,
    method: str = THRESHOLD_CALIBRATION_METHOD,
    threshold_percentile: float = AE_THRESHOLD_PERCENTILE,
    target_fpr: float = THRESHOLD_TARGET_FPR,
    n_bootstrap: int = THRESHOLD_N_BOOTSTRAP,
    n_candidates: int = THRESHOLD_N_CANDIDATES,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Punto unico di selezione della soglia. Ritorna un dict con la soglia
    scelta e diagnostica (FPR/TPR raggiunti sul calibration set, J-stat,
    deviazione standard delle soglie bootstrap).

    Vedi il blocco di commenti "NOVITA' v3" in CONFIGURATION per la
    spiegazione dei singoli metodi.
    """
    if method == "percentile" or err_anomaly is None or len(err_anomaly) == 0:
        thr = float(np.percentile(err_real, threshold_percentile))
        fpr = float(np.mean(err_real > thr))
        return {
            "threshold": thr, "method": "percentile",
            "j_stat": None, "achieved_fpr": fpr, "achieved_tpr": None,
            "bootstrap_threshold_std": None,
        }

    combined   = np.concatenate([err_real, err_anomaly])
    candidates = np.unique(np.percentile(combined, np.linspace(0, 100, n_candidates)))

    if method == "youden":
        thr, j = _best_threshold_on_sample(err_real, err_anomaly, candidates, "youden", target_fpr)
        fpr = float(np.mean(err_real > thr))
        tpr = float(np.mean(err_anomaly > thr))
        return {
            "threshold": thr, "method": "youden", "j_stat": j,
            "achieved_fpr": fpr, "achieved_tpr": tpr,
            "bootstrap_threshold_std": None,
        }

    if method in ("youden_robust", "fpr_constrained"):
        mode = "youden" if method == "youden_robust" else "fpr_constrained"
        rng  = np.random.RandomState(random_state)
        n_real, n_anom = len(err_real), len(err_anomaly)

        boot_thrs = []
        for _ in range(n_bootstrap):
            idx_r = rng.randint(0, n_real, n_real)
            idx_a = rng.randint(0, n_anom, n_anom)
            thr, _ = _best_threshold_on_sample(
                err_real[idx_r], err_anomaly[idx_a], candidates, mode, target_fpr
            )
            boot_thrs.append(thr)

        thr = float(np.median(boot_thrs))
        fpr = float(np.mean(err_real > thr))
        tpr = float(np.mean(err_anomaly > thr))
        return {
            "threshold": thr, "method": method, "j_stat": tpr - fpr,
            "achieved_fpr": fpr, "achieved_tpr": tpr,
            "bootstrap_threshold_std": float(np.std(boot_thrs)),
        }

    raise ValueError(f"THRESHOLD_CALIBRATION_METHOD sconosciuto: {method}")


# ==============================================================================
# 4. ONE-CLASS AUTOENCODER (sklearn-compatibile)
# ==============================================================================

class OneClassAutoencoder(BaseEstimator):
    """
    One-Class Autoencoder per anomaly/novelty detection.

    Interfaccia sklearn-compatibile (supports Pipeline, GridSearchCV, clone):
      fit(X)               -> addestra la rete, calcola una soglia di fallback
      predict(X)           -> +1 (inlier) se MSE <= soglia, -1 (outlier) altrimenti
      decision_function(X) -> threshold_ - MSE(X)  (>= 0 = inlier, < 0 = outlier)
      score_samples(X)     -> alias di decision_function
      calibrate_threshold(...) -> ricalibra la soglia DOPO il fit (vedi
                                   _select_threshold per i metodi disponibili)
    """

    def __init__(
        self,
        hidden_dims: tuple = AE_HIDDEN_DIMS,
        latent_dim: int = AE_LATENT_DIM,
        learning_rate: float = AE_LEARNING_RATE,
        epochs: int = AE_EPOCHS,
        batch_size: int = AE_BATCH_SIZE,
        dropout_rate: float = AE_DROPOUT_RATE,
        activation: str = AE_ACTIVATION,
        threshold_percentile: float = AE_THRESHOLD_PERCENTILE,
        weight_decay: float = AE_WEIGHT_DECAY,
        early_stopping: bool = AE_EARLY_STOPPING,
        patience: int = AE_PATIENCE,
        use_gpu: bool = AE_USE_GPU,
        random_state: int = 42,
    ):
        self.hidden_dims          = hidden_dims
        self.latent_dim           = latent_dim
        self.learning_rate        = learning_rate
        self.epochs               = epochs
        self.batch_size           = batch_size
        self.dropout_rate         = dropout_rate
        self.activation           = activation
        self.threshold_percentile = threshold_percentile
        self.weight_decay         = weight_decay
        self.early_stopping       = early_stopping
        self.patience             = patience
        self.use_gpu              = use_gpu
        self.random_state         = random_state

    def _get_device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _mse_errors(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
        self.net_.eval()
        with torch.no_grad():
            recon  = self.net_(X_t)
            errors = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def fit(self, X, y=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        hidden_dims = tuple(self.hidden_dims)
        device      = self._get_device()
        self.device_    = device
        self.input_dim_ = X.shape[1]

        use_holdout = len(X) >= 60
        if use_holdout:
            X_fit, X_holdout = train_test_split(
                X, test_size=0.2, random_state=self.random_state, shuffle=True
            )
            X_es, X_thresh = train_test_split(
                X_holdout, test_size=0.5, random_state=self.random_state, shuffle=True
            )
        else:
            X_fit, X_es, X_thresh = X, X, X

        X_t     = torch.tensor(X_fit, dtype=torch.float32).to(device)
        X_es_t  = torch.tensor(X_es, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_t)
        loader  = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=(len(dataset) > self.batch_size),
        )

        self.net_ = _AutoencoderNet(
            self.input_dim_, hidden_dims, self.latent_dim,
            self.dropout_rate, self.activation,
        ).to(device)

        optimizer = torch.optim.Adam(
            self.net_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        self.train_losses_ = []
        self.val_losses_   = []
        best_loss          = float("inf")
        patience_counter   = 0
        best_state         = None

        for epoch in range(self.epochs):
            self.net_.train()
            epoch_loss = 0.0
            n_seen     = 0
            for (batch,) in loader:
                if batch.shape[0] < 2:
                    continue
                optimizer.zero_grad()
                recon = self.net_(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
                n_seen     += len(batch)

            mean_loss = epoch_loss / max(n_seen, 1)
            self.train_losses_.append(mean_loss)

            self.net_.eval()
            with torch.no_grad():
                recon_es = self.net_(X_es_t)
                val_loss = criterion(recon_es, X_es_t).item()
            self.val_losses_.append(val_loss)

            monitor_loss = val_loss if use_holdout else mean_loss

            if self.early_stopping:
                if monitor_loss < best_loss - 1e-8:
                    best_loss        = monitor_loss
                    patience_counter = 0
                    best_state       = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if best_state is not None:
                            self.net_.load_state_dict(best_state)
                        break

        self.n_epochs_trained_ = len(self.train_losses_)

        threshold_errors = self._mse_errors(X_thresh)
        self.threshold_        = float(np.percentile(threshold_errors, self.threshold_percentile))
        self.threshold_method_ = "percentile"
        # Diagnostica di calibrazione: popolata solo dopo calibrate_threshold()
        self.calibration_j_stat_        = None
        self.calibration_achieved_fpr_  = None
        self.calibration_achieved_tpr_  = None
        self.calibration_threshold_std_ = None

        return self

    def decision_function(self, X) -> np.ndarray:
        return self.threshold_ - self._mse_errors(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int32)

    def score_samples(self, X) -> np.ndarray:
        return self.decision_function(X)

    def calibrate_threshold(
        self,
        X_real_calib,
        X_anomaly_calib=None,
        method: str = THRESHOLD_CALIBRATION_METHOD,
        target_fpr: float = THRESHOLD_TARGET_FPR,
        n_bootstrap: int = THRESHOLD_N_BOOTSTRAP,
        n_candidates: int = THRESHOLD_N_CANDIDATES,
    ):
        """
        Ricalibra self.threshold_ DOPO il training, senza toccare i pesi
        della rete. Vedi _select_threshold per la spiegazione dei metodi
        disponibili ("percentile" | "youden" | "youden_robust" |
        "fpr_constrained").
        """
        err_real    = self._mse_errors(X_real_calib)
        err_anomaly = (
            self._mse_errors(X_anomaly_calib)
            if X_anomaly_calib is not None and len(X_anomaly_calib) > 0
            else None
        )

        result = _select_threshold(
            err_real, err_anomaly, method=method,
            threshold_percentile=self.threshold_percentile,
            target_fpr=target_fpr, n_bootstrap=n_bootstrap, n_candidates=n_candidates,
            random_state=self.random_state,
        )

        self.threshold_                 = result["threshold"]
        self.threshold_method_          = result["method"]
        self.calibration_j_stat_        = result["j_stat"]
        self.calibration_achieved_fpr_  = result["achieved_fpr"]
        self.calibration_achieved_tpr_  = result["achieved_tpr"]
        self.calibration_threshold_std_ = result["bootstrap_threshold_std"]
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["_net_cpu_state_"] = {
                k: v.cpu().clone() for k, v in state["net_"].state_dict().items()
            }
            del state["net_"]
        state.pop("device_", None)
        return state

    def __setstate__(self, state):
        net_cpu_state = state.pop("_net_cpu_state_", None)
        self.__dict__.update(state)
        self.device_ = torch.device("cpu")
        if net_cpu_state is not None:
            self.net_ = _AutoencoderNet(
                self.input_dim_,
                tuple(self.hidden_dims),
                self.latent_dim,
                self.dropout_rate,
                self.activation,
            )
            self.net_.load_state_dict(net_cpu_state)
            self.net_.eval()


# ==============================================================================
# 4b. ENSEMBLE DI ONE-CLASS AUTOENCODER (bagging)
# ==============================================================================

class OneClassAutoencoderEnsemble(BaseEstimator):
    """
    Ensemble (bagging) di piu' OneClassAutoencoder indipendenti. Espone la
    stessa interfaccia sklearn-compatibile di OneClassAutoencoder.
    """

    def __init__(
        self,
        n_estimators: int = AE_ENSEMBLE_SIZE,
        bootstrap: bool = AE_ENSEMBLE_BOOTSTRAP,
        hidden_dims: tuple = AE_HIDDEN_DIMS,
        latent_dim: int = AE_LATENT_DIM,
        learning_rate: float = AE_LEARNING_RATE,
        epochs: int = AE_EPOCHS,
        batch_size: int = AE_BATCH_SIZE,
        dropout_rate: float = AE_DROPOUT_RATE,
        activation: str = AE_ACTIVATION,
        threshold_percentile: float = AE_THRESHOLD_PERCENTILE,
        weight_decay: float = AE_WEIGHT_DECAY,
        early_stopping: bool = AE_EARLY_STOPPING,
        patience: int = AE_PATIENCE,
        use_gpu: bool = AE_USE_GPU,
        random_state: int = 42,
    ):
        self.n_estimators         = n_estimators
        self.bootstrap             = bootstrap
        self.hidden_dims           = hidden_dims
        self.latent_dim            = latent_dim
        self.learning_rate         = learning_rate
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.dropout_rate          = dropout_rate
        self.activation            = activation
        self.threshold_percentile  = threshold_percentile
        self.weight_decay          = weight_decay
        self.early_stopping        = early_stopping
        self.patience              = patience
        self.use_gpu               = use_gpu
        self.random_state          = random_state

    def fit(self, X, y=None):
        self.members_ = []
        for i in range(self.n_estimators):
            seed = self.random_state + i
            X_member = (
                resample(X, replace=True, n_samples=len(X), random_state=seed)
                if self.bootstrap else X
            )
            member = OneClassAutoencoder(
                hidden_dims           = self.hidden_dims,
                latent_dim            = self.latent_dim,
                learning_rate         = self.learning_rate,
                epochs                = self.epochs,
                batch_size            = self.batch_size,
                dropout_rate          = self.dropout_rate,
                activation            = self.activation,
                threshold_percentile  = self.threshold_percentile,
                weight_decay          = self.weight_decay,
                early_stopping        = self.early_stopping,
                patience              = self.patience,
                use_gpu               = self.use_gpu,
                random_state          = seed,
            )
            member.fit(X_member)
            self.members_.append(member)

        self.threshold_        = float(np.mean([m.threshold_ for m in self.members_]))
        self.threshold_method_ = "percentile"
        self.calibration_j_stat_        = None
        self.calibration_achieved_fpr_  = None
        self.calibration_achieved_tpr_  = None
        self.calibration_threshold_std_ = None
        return self

    def _mean_mse(self, X) -> np.ndarray:
        errors = np.stack([m._mse_errors(X) for m in self.members_], axis=0)
        return errors.mean(axis=0)

    def decision_function(self, X) -> np.ndarray:
        return self.threshold_ - self._mean_mse(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int32)

    def score_samples(self, X) -> np.ndarray:
        return self.decision_function(X)

    def calibrate_threshold(
        self,
        X_real_calib,
        X_anomaly_calib=None,
        method: str = THRESHOLD_CALIBRATION_METHOD,
        target_fpr: float = THRESHOLD_TARGET_FPR,
        n_bootstrap: int = THRESHOLD_N_BOOTSTRAP,
        n_candidates: int = THRESHOLD_N_CANDIDATES,
    ):
        """Stessa logica di OneClassAutoencoder.calibrate_threshold, ma sull'errore medio dell'ensemble."""
        err_real    = self._mean_mse(X_real_calib)
        err_anomaly = (
            self._mean_mse(X_anomaly_calib)
            if X_anomaly_calib is not None and len(X_anomaly_calib) > 0
            else None
        )

        result = _select_threshold(
            err_real, err_anomaly, method=method,
            threshold_percentile=self.threshold_percentile,
            target_fpr=target_fpr, n_bootstrap=n_bootstrap, n_candidates=n_candidates,
            random_state=self.random_state,
        )

        self.threshold_                 = result["threshold"]
        self.threshold_method_          = result["method"]
        self.calibration_j_stat_        = result["j_stat"]
        self.calibration_achieved_fpr_  = result["achieved_fpr"]
        self.calibration_achieved_tpr_  = result["achieved_tpr"]
        self.calibration_threshold_std_ = result["bootstrap_threshold_std"]
        return self


# ==============================================================================
# 4c. HELPER: DESCRIZIONE CLASSIFICATORE (single o ensemble) PER LOG/METRICHE
# ==============================================================================

def _classifier_info(clf) -> dict:
    is_ensemble = hasattr(clf, "members_")
    if is_ensemble:
        members = clf.members_
        info = {
            "type":                      "ensemble",
            "n_estimators":              clf.n_estimators,
            "bootstrap":                 clf.bootstrap,
            "hidden_dims":               list(clf.hidden_dims),
            "latent_dim":                clf.latent_dim,
            "learning_rate":             clf.learning_rate,
            "epochs_max":                clf.epochs,
            "epochs_trained":            None,
            "epochs_trained_per_member": [int(m.n_epochs_trained_) for m in members],
            "batch_size":                clf.batch_size,
            "dropout_rate":              clf.dropout_rate,
            "activation":                clf.activation,
            "weight_decay":              clf.weight_decay,
            "early_stopping":            clf.early_stopping,
            "patience":                  clf.patience,
        }
    else:
        info = {
            "type":                      "single",
            "hidden_dims":               list(clf.hidden_dims),
            "latent_dim":                clf.latent_dim,
            "learning_rate":             clf.learning_rate,
            "epochs_max":                clf.epochs,
            "epochs_trained":            int(clf.n_epochs_trained_) if hasattr(clf, "n_epochs_trained_") else None,
            "epochs_trained_per_member": None,
            "batch_size":                clf.batch_size,
            "dropout_rate":              clf.dropout_rate,
            "activation":                clf.activation,
            "weight_decay":              clf.weight_decay,
            "early_stopping":            clf.early_stopping,
            "patience":                  clf.patience,
        }

    info["threshold_percentile"]      = clf.threshold_percentile
    info["threshold_mse"]             = float(clf.threshold_) if hasattr(clf, "threshold_") else None
    info["threshold_method"]          = getattr(clf, "threshold_method_", "percentile")
    info["calibration_j_stat"]        = getattr(clf, "calibration_j_stat_", None)
    info["calibration_achieved_fpr"]  = getattr(clf, "calibration_achieved_fpr_", None)
    info["calibration_achieved_tpr"]  = getattr(clf, "calibration_achieved_tpr_", None)
    info["calibration_threshold_std"] = getattr(clf, "calibration_threshold_std_", None)
    return info


# ==============================================================================
# 5. PIPELINE (Scaler + [PCA] + OneClassAutoencoder / Ensemble)
# ==============================================================================

def build_pipeline(use_ensemble: bool = True) -> Pipeline:
    steps = []
    if SCALE_FEATURES:
        scaler = RobustScaler() if SCALER_TYPE == "robust" else StandardScaler()
        steps.append(("scaler", scaler))
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)))

    common_kwargs = dict(
        hidden_dims=AE_HIDDEN_DIMS,
        latent_dim=AE_LATENT_DIM,
        learning_rate=AE_LEARNING_RATE,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        dropout_rate=AE_DROPOUT_RATE,
        activation=AE_ACTIVATION,
        threshold_percentile=AE_THRESHOLD_PERCENTILE,
        weight_decay=AE_WEIGHT_DECAY,
        early_stopping=AE_EARLY_STOPPING,
        patience=AE_PATIENCE,
        use_gpu=AE_USE_GPU,
        random_state=RANDOM_STATE,
    )

    if use_ensemble and AE_ENSEMBLE_SIZE > 1:
        classifier = OneClassAutoencoderEnsemble(
            n_estimators=AE_ENSEMBLE_SIZE,
            bootstrap=AE_ENSEMBLE_BOOTSTRAP,
            **common_kwargs,
        )
    else:
        classifier = OneClassAutoencoder(**common_kwargs)

    steps.append(("classifier", classifier))
    return Pipeline(steps)


# ==============================================================================
# 5b. CALIBRAZIONE SOGLIA A LIVELLO DI PIPELINE
# ==============================================================================

def _transform_features(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    Xt = X
    for _, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def split_calibration_set(
    X_real_tv: np.ndarray,
    cloned_tv: dict,
    calibration_size: float = CALIBRATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    X_real_fit, X_real_calib = train_test_split(
        X_real_tv, test_size=calibration_size, random_state=random_state, shuffle=True
    )
    cloned_fit, cloned_calib = {}, {}
    for name, X in cloned_tv.items():
        X_fit, X_calib = train_test_split(
            X, test_size=calibration_size, random_state=random_state, shuffle=True
        )
        cloned_fit[name]   = X_fit
        cloned_calib[name] = X_calib

    log.info(f"\nCalibration set riservato ({calibration_size:.0%} di train/val, mai usato per il training):")
    log.info(f"  Reali:  fit={len(X_real_fit)}  |  calib={len(X_real_calib)}")
    for name in cloned_fit:
        log.info(f"  [{name:>12}]: fit={len(cloned_fit[name])}  |  calib={len(cloned_calib[name])}")

    return X_real_fit, X_real_calib, cloned_fit, cloned_calib


def calibrate_pipeline_threshold(
    pipe: Pipeline,
    X_real_calib: np.ndarray,
    cloned_calib: dict,
    method: str = THRESHOLD_CALIBRATION_METHOD,
    target_fpr: float = THRESHOLD_TARGET_FPR,
    n_bootstrap: int = THRESHOLD_N_BOOTSTRAP,
    n_candidates: int = THRESHOLD_N_CANDIDATES,
) -> Pipeline:
    """
    Ricalibra la soglia del classificatore finale della pipeline usando il
    calibration set (mai visto prima). Le feature vengono trasformate con gli
    step di preprocessing gia' fittati (scaler/pca), senza rifittarli.
    """
    if X_real_calib is None or len(X_real_calib) == 0:
        log.warning("  Calibration set vuoto: soglia lasciata al valore di fallback (percentile).")
        return pipe

    X_real_calib_t = _transform_features(pipe, X_real_calib)
    X_anomaly_calib_t = None
    if cloned_calib:
        X_anomaly_calib = np.vstack(list(cloned_calib.values()))
        X_anomaly_calib_t = _transform_features(pipe, X_anomaly_calib)

    clf = pipe.named_steps["classifier"]
    clf.calibrate_threshold(
        X_real_calib_t, X_anomaly_calib_t, method=method,
        target_fpr=target_fpr, n_bootstrap=n_bootstrap, n_candidates=n_candidates,
    )

    info    = _classifier_info(clf)
    j_str   = f"  |  J-stat={info['calibration_j_stat']:.4f}" if info["calibration_j_stat"] is not None else ""
    fpr_str = f"  |  FPR calib={info['calibration_achieved_fpr']:.4f}" if info["calibration_achieved_fpr"] is not None else ""
    tpr_str = f"  |  TPR calib={info['calibration_achieved_tpr']:.4f}" if info["calibration_achieved_tpr"] is not None else ""
    std_str = (
        f"  |  std soglia (bootstrap)={info['calibration_threshold_std']:.6f}"
        if info["calibration_threshold_std"] is not None else ""
    )
    log.info(
        f"\n  Soglia ricalibrata sul calibration set (metodo={info['threshold_method']}, "
        f"target_fpr={target_fpr:.2f}): {info['threshold_mse']:.6f}{j_str}{fpr_str}{tpr_str}{std_str}"
    )
    return pipe


# ==============================================================================
# 6. PAIRED ONE-CLASS K-FOLD ITERATOR
# ==============================================================================

class PairedOneClassKFold:
    def __init__(
        self,
        n_real: int,
        n_cloned: int,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_real       = n_real
        self.n_cloned     = n_cloned
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        real_pos   = np.arange(self.n_real)
        cloned_pos = np.arange(self.n_cloned)

        real_splits   = list(kf.split(real_pos))
        cloned_splits = list(kf.split(cloned_pos))

        for (r_train, r_val), (_, c_val) in zip(real_splits, cloned_splits):
            train_idx = real_pos[r_train]
            val_idx   = np.concatenate([
                real_pos[r_val],
                self.n_real + cloned_pos[c_val],
            ])
            yield train_idx, val_idx


# ==============================================================================
# 7. CUSTOM SCORER PAIRED (balanced accuracy a soglia fissa + ROC-AUC rank-based)
# ==============================================================================

def balanced_scorer(estimator, X, y):
    real_mask   = (y == 0)
    cloned_mask = (y == 1)

    y_pred = estimator.predict(X)

    inlier_rate    = float(np.mean(y_pred[real_mask]   ==  1)) if real_mask.any()   else 1.0
    detection_rate = float(np.mean(y_pred[cloned_mask] == -1)) if cloned_mask.any() else 0.0

    return (inlier_rate + detection_rate) / 2.0


def roc_auc_paired_scorer(estimator, X, y):
    scores = estimator.decision_function(X)
    anomaly_scores = -scores
    try:
        return float(roc_auc_score(y, anomaly_scores))
    except ValueError:
        return 0.5


# ==============================================================================
# 8. GRID / RANDOMIZED SEARCH (con PairedOneClassKFold)
# ==============================================================================

def run_grid_search(X_real_tv: np.ndarray, cloned_tv: dict) -> dict:
    X_all_cloned_tv = np.vstack(list(cloned_tv.values()))
    n_real   = len(X_real_tv)
    n_cloned = len(X_all_cloned_tv)

    X_combined = np.vstack([X_real_tv, X_all_cloned_tv])
    y_combined = np.concatenate([
        np.zeros(n_real,  dtype=np.float32),
        np.ones(n_cloned, dtype=np.float32),
    ])

    total_combinations = _count_combinations(GRID_PARAM_GRID)
    n_iter      = GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else total_combinations
    search_type = "RandomizedSearchCV" if GRID_SEARCH_RANDOMIZED else "GridSearchCV"
    total_fits  = n_iter * GRID_SEARCH_CV_FOLDS

    use_cuda = AE_USE_GPU and torch.cuda.is_available()
    n_jobs   = 1 if use_cuda else GRID_SEARCH_N_JOBS
    if use_cuda and GRID_SEARCH_N_JOBS != 1:
        log.warning("  GPU rilevata: n_jobs forzato a 1 per compatibilita' CUDA.")

    scoring_fn = roc_auc_paired_scorer if GRID_SEARCH_USE_AUC_SCORING else balanced_scorer

    log.info(f"\n{'=' * 60}")
    log.info(f"{search_type} con PairedOneClassKFold")
    log.info(f"Scoring: {GRID_SEARCH_SCORING}  |  CV folds: {GRID_SEARCH_CV_FOLDS}")
    log.info(f"Train fold: solo reali  |  Val fold: reali_val + clonati_val")
    log.info(f"Reali train/val: {n_real}  |  Clonati train/val (tot): {n_cloned}")
    log.info(f"Combinazioni spazio: {total_combinations}  |  Testate: {n_iter}  |  Fit totali: {total_fits}")
    log.info(f"Device: {'CUDA' if use_cuda else 'CPU'}  |  n_jobs: {n_jobs}")
    log.info(f"{'=' * 60}")

    base_pipe = build_pipeline(use_ensemble=False)
    cv = PairedOneClassKFold(
        n_real=n_real,
        n_cloned=n_cloned,
        n_splits=GRID_SEARCH_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    log.info(f"Avvio ricerca iperparametri ({total_fits} fit) -- puo' richiedere diversi minuti...")

    if GRID_SEARCH_RANDOMIZED:
        searcher = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=GRID_PARAM_GRID,
            n_iter=GRID_SEARCH_N_ITER,
            scoring=scoring_fn,
            cv=cv,
            n_jobs=n_jobs,
            verbose=3,
            refit=False,
            random_state=RANDOM_STATE,
        )
    else:
        searcher = GridSearchCV(
            estimator=base_pipe,
            param_grid=GRID_PARAM_GRID,
            scoring=scoring_fn,
            cv=cv,
            n_jobs=n_jobs,
            verbose=3,
            refit=False,
        )

    searcher.fit(X_combined, y_combined)
    log.info("  Ricerca completata.")
    log.info(f"\n  Best score ({GRID_SEARCH_SCORING}): {searcher.best_score_:.4f}")
    log.info(f"  Best params: {searcher.best_params_}")

    final_pipe = build_pipeline(use_ensemble=(AE_ENSEMBLE_SIZE > 1))
    final_pipe.set_params(**searcher.best_params_)
    final_pipe.fit(X_real_tv)
    log.info("  Pipeline finale (ensemble se AE_ENSEMBLE_SIZE>1) fittata su X_real_tv (soli reali -- one-class corretto).")

    clf  = final_pipe.named_steps["classifier"]
    info = _classifier_info(clf)
    if info["type"] == "ensemble":
        log.info(
            f"  Ensemble di {info['n_estimators']} membri  |  "
            f"Epoche per membro: {info['epochs_trained_per_member']}  |  "
            f"Soglia MSE (fallback): {info['threshold_mse']:.6f}"
        )
    else:
        log.info(
            f"  Epoche effettive: {info['epochs_trained']}  |  "
            f"Soglia MSE (fallback): {info['threshold_mse']:.6f}"
        )

    gs_results = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
    gs_results_path = METRICS_DIR / "grid_search_results.csv"
    gs_results.to_csv(gs_results_path, index=False)
    log.info(f"  Risultati grid search salvati: {gs_results_path}")

    return {
        "best_params":   searcher.best_params_,
        "best_score":    float(searcher.best_score_),
        "best_pipeline": final_pipe,
    }


def _count_combinations(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


# ==============================================================================
# 9. K-FOLD CV PAIRED (su train/val -- diagnostica out-of-sample)
# ==============================================================================

def run_kfold(X_real_tv: np.ndarray, cloned_tv: dict, best_pipeline=None) -> dict:
    log.info(f"\n{'=' * 60}")
    log.info(f"K-Fold CV paired (k={N_FOLDS}) su train/val -- diagnostica out-of-sample")
    if best_pipeline is not None:
        log.info("  Usando i best params dalla Grid Search")
    log.info(f"{'=' * 60}")

    import sklearn

    X_all_cloned_tv = np.vstack(list(cloned_tv.values()))
    n_real   = len(X_real_tv)
    n_cloned = len(X_all_cloned_tv)

    X_combined = np.vstack([X_real_tv, X_all_cloned_tv])
    y_combined = np.concatenate([
        np.zeros(n_real,  dtype=int),
        np.ones(n_cloned, dtype=int),
    ])

    kf = PairedOneClassKFold(
        n_real=n_real,
        n_cloned=n_cloned,
        n_splits=N_FOLDS,
        shuffle=SHUFFLE_FOLDS,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined, y_combined), start=1):
        X_tr = X_combined[train_idx]
        X_vl = X_combined[val_idx]
        y_vl = y_combined[val_idx]

        pipe = sklearn.clone(best_pipeline) if best_pipeline is not None else build_pipeline()
        pipe.fit(X_tr)

        y_pred  = pipe.predict(X_vl)
        df_vals = pipe.decision_function(X_vl)

        real_mask   = (y_vl == 0)
        cloned_mask = (y_vl == 1)

        inlier_rate    = float(np.mean(y_pred[real_mask]   ==  1)) if real_mask.any()   else np.nan
        detection_rate = float(np.mean(y_pred[cloned_mask] == -1)) if cloned_mask.any() else np.nan
        fpr_fold  = 1.0 - inlier_rate    if not np.isnan(inlier_rate)    else np.nan
        fnr_fold  = 1.0 - detection_rate if not np.isnan(detection_rate) else np.nan
        balanced  = (
            (inlier_rate + detection_rate) / 2.0
            if not (np.isnan(inlier_rate) or np.isnan(detection_rate))
            else np.nan
        )
        try:
            auc_fold = float(roc_auc_score(y_vl, -df_vals))
        except ValueError:
            auc_fold = np.nan

        clf  = pipe.named_steps.get("classifier")
        info = _classifier_info(clf) if clf is not None else {}
        fold_metrics.append({
            "fold":                  fold,
            "n_train_real":          int(len(X_tr)),
            "n_val_real":            int(np.sum(real_mask)),
            "n_val_cloned":          int(np.sum(cloned_mask)),
            "inlier_rate_real":      inlier_rate,
            "false_positive_rate":   fpr_fold,
            "detection_rate_cloned": detection_rate,
            "false_negative_rate":   fnr_fold,
            "balanced_score":        balanced,
            "roc_auc":               auc_fold,
            "mean_decision_score":   float(np.mean(df_vals)),
            "n_epochs_trained":      info.get("epochs_trained"),
            "threshold_mse":         info.get("threshold_mse"),
        })
        log.info(
            f"  Fold {fold}: "
            f"Inlier(real)={inlier_rate:.4f}  FPR={fpr_fold:.4f}  |  "
            f"Detect(cloned)={detection_rate:.4f}  FNR={fnr_fold:.4f}  |  "
            f"Balanced={balanced:.4f}  |  AUC={auc_fold:.4f}"
        )

    def _mean_std(key):
        vals = [m[key] for m in fold_metrics if m.get(key) is not None and not np.isnan(m[key])]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)

    mean_inlier, std_inlier = _mean_std("inlier_rate_real")
    mean_fpr,    std_fpr    = _mean_std("false_positive_rate")
    mean_det,    std_det    = _mean_std("detection_rate_cloned")
    mean_fnr,    std_fnr    = _mean_std("false_negative_rate")
    mean_bal,    std_bal    = _mean_std("balanced_score")
    mean_auc,    std_auc    = _mean_std("roc_auc")

    log.info(f"\n  Media Inlier Rate (reali)       : {mean_inlier:.4f} +/- {std_inlier:.4f}")
    log.info(f"  Media False Positive Rate       : {mean_fpr:.4f}    +/- {std_fpr:.4f}")
    log.info(f"  Media Detection Rate (clonati)  : {mean_det:.4f}    +/- {std_det:.4f}")
    log.info(f"  Media False Negative Rate       : {mean_fnr:.4f}    +/- {std_fnr:.4f}")
    log.info(f"  Media Balanced Score            : {mean_bal:.4f}    +/- {std_bal:.4f}")
    log.info(f"  Media ROC-AUC                   : {mean_auc:.4f}    +/- {std_auc:.4f}")

    return {
        "folds":                    fold_metrics,
        "mean_inlier_rate":         mean_inlier,
        "std_inlier_rate":          std_inlier,
        "mean_false_positive_rate": mean_fpr,
        "std_false_positive_rate":  std_fpr,
        "mean_detection_rate":      mean_det,
        "std_detection_rate":       std_det,
        "mean_false_negative_rate": mean_fnr,
        "std_false_negative_rate":  std_fnr,
        "mean_balanced_score":      mean_bal,
        "std_balanced_score":       std_bal,
        "mean_roc_auc":             mean_auc,
        "std_roc_auc":              std_auc,
    }


# ==============================================================================
# 10. ADDESTRAMENTO FINALE, CALIBRAZIONE SOGLIA E VALUTAZIONE SUL TEST SET
# ==============================================================================

def train_and_evaluate(
    X_real_tv: np.ndarray,
    X_real_test: np.ndarray,
    cloned_tv: dict,
    cloned_test: dict,
    best_pipeline=None,
    X_real_calib: np.ndarray = None,
    cloned_calib: dict = None,
) -> tuple:
    log.info(f"\n{'=' * 60}")
    log.info("Addestramento finale su X_real_tv + valutazione sul TEST SET")
    log.info(f"{'=' * 60}")

    if best_pipeline is not None:
        pipe = best_pipeline
        log.info("  Usando la pipeline fittata su X_real_tv dalla Grid Search.")
    else:
        pipe = build_pipeline()
        pipe.fit(X_real_tv)
        log.info("  Fitting con parametri di default (Grid Search disabilitato).")

    clf = pipe.named_steps.get("classifier")
    if clf is not None:
        info = _classifier_info(clf)
        if info["type"] == "ensemble":
            log.info(
                f"  Ensemble di {info['n_estimators']} membri  |  "
                f"Epoche per membro: {info['epochs_trained_per_member']}  |  "
                f"Soglia MSE (fallback): {info['threshold_mse']:.6f}"
            )
        else:
            log.info(
                f"  Epoche effettive: {info['epochs_trained']}  |  "
                f"Soglia MSE (fallback): {info['threshold_mse']:.6f}"
            )

    if X_real_calib is not None:
        pipe = calibrate_pipeline_threshold(
            pipe, X_real_calib, cloned_calib,
            method=THRESHOLD_CALIBRATION_METHOD,
            target_fpr=THRESHOLD_TARGET_FPR,
            n_bootstrap=THRESHOLD_N_BOOTSTRAP,
            n_candidates=THRESHOLD_N_CANDIDATES,
        )

    log.info(f"  Campioni training (real tv): {len(X_real_tv)}")
    log.info(f"  Campioni test (real):        {len(X_real_test)}")
    for name in cloned_test:
        log.info(f"  Campioni test [{name:>12}]:   {len(cloned_test[name])}")

    y_pred_real_test  = pipe.predict(X_real_test)
    df_vals_real_test = pipe.decision_function(X_real_test)

    inlier_rate_test = float(np.mean(y_pred_real_test == 1))
    fpr_test         = 1.0 - inlier_rate_test

    log.info(f"\n  [TEST SET -- AUDIO REALI]")
    log.info(f"  Inlier rate: {inlier_rate_test:.4f}  |  FPR: {fpr_test:.4f}")
    log.info(f"  Predicted inlier: {int(np.sum(y_pred_real_test == 1))}/{len(y_pred_real_test)}")
    log.info(f"  Mean decision score: {np.mean(df_vals_real_test):.4f}")

    per_source_results = {}
    per_source_df_vals = {}

    for name, X_cloned_test in cloned_test.items():
        y_pred_cloned_test  = pipe.predict(X_cloned_test)
        df_vals_cloned_test = pipe.decision_function(X_cloned_test)

        detection_rate = float(np.mean(y_pred_cloned_test == -1))
        fnr            = 1.0 - detection_rate

        y_true_src     = np.concatenate([
            np.zeros(len(X_real_test),  dtype=int),
            np.ones(len(X_cloned_test), dtype=int),
        ])
        y_pred_raw_src = np.concatenate([y_pred_real_test, y_pred_cloned_test])
        y_pred_src     = np.where(y_pred_raw_src == 1, 0, 1)
        anomaly_scores = np.concatenate([-df_vals_real_test, -df_vals_cloned_test])

        acc_src = accuracy_score(y_true_src, y_pred_src)
        auc_src = roc_auc_score(y_true_src, anomaly_scores)
        cm_src  = confusion_matrix(y_true_src, y_pred_src)
        rep_src = classification_report(
            y_true_src, y_pred_src,
            target_names=[LABEL_NORMAL, LABEL_ANOMALY],
            output_dict=True,
        )

        per_source_results[name] = {
            "n_samples":              int(len(X_cloned_test)),
            "anomaly_detection_rate": detection_rate,
            "false_negative_rate":    fnr,
            "n_predicted_outlier":    int(np.sum(y_pred_cloned_test == -1)),
            "n_predicted_inlier":     int(np.sum(y_pred_cloned_test == 1)),
            "mean_decision_score":    float(np.mean(df_vals_cloned_test)),
            "accuracy_vs_real_test":  float(acc_src),
            "roc_auc_vs_real_test":   float(auc_src),
            "confusion_matrix":       cm_src.tolist(),
            "classification_report":  rep_src,
        }
        per_source_df_vals[name] = df_vals_cloned_test

        log.info(f"\n  [TEST -- {name.upper()}]")
        log.info(f"  Anomaly detection rate: {detection_rate:.4f}  |  FNR: {fnr:.4f}")
        log.info(f"  Accuracy (vs real test): {acc_src:.4f}  |  ROC-AUC: {auc_src:.4f}")
        log.info(
            f"\n{classification_report(y_true_src, y_pred_src, target_names=[LABEL_NORMAL, LABEL_ANOMALY])}"
        )

        _save_confusion_matrix(
            cm_src, f"test_real_vs_{name}",
            f"Confusion Matrix (Test Set) -- Real vs {name.capitalize()}"
        )

    X_all_cloned_test      = np.vstack(list(cloned_test.values()))
    y_pred_all_cloned_test  = pipe.predict(X_all_cloned_test)
    df_vals_all_cloned_test = pipe.decision_function(X_all_cloned_test)

    detection_rate_all = float(np.mean(y_pred_all_cloned_test == -1))
    fnr_all            = 1.0 - detection_rate_all

    y_true_comb    = np.concatenate([
        np.zeros(len(X_real_test),      dtype=int),
        np.ones(len(X_all_cloned_test), dtype=int),
    ])
    y_pred_raw_comb = np.concatenate([y_pred_real_test, y_pred_all_cloned_test])
    y_pred_comb     = np.where(y_pred_raw_comb == 1, 0, 1)
    anomaly_scores_comb = np.concatenate([-df_vals_real_test, -df_vals_all_cloned_test])

    acc_comb = accuracy_score(y_true_comb, y_pred_comb)
    auc_comb = roc_auc_score(y_true_comb, anomaly_scores_comb)
    cm_comb  = confusion_matrix(y_true_comb, y_pred_comb)
    rep_comb = classification_report(
        y_true_comb, y_pred_comb,
        target_names=[LABEL_NORMAL, LABEL_ANOMALY],
        output_dict=True,
    )

    sources_str = " + ".join(cloned_test.keys())
    log.info(f"\n  [TEST COMBINATO: real vs {sources_str}]")
    log.info(f"  Clonati test totali: {len(X_all_cloned_test)}")
    log.info(f"  Anomaly detection rate: {detection_rate_all:.4f}  |  FNR: {fnr_all:.4f}")
    log.info(f"  Accuracy: {acc_comb:.4f}  |  ROC-AUC: {auc_comb:.4f}")
    log.info(
        f"\n{classification_report(y_true_comb, y_pred_comb, target_names=[LABEL_NORMAL, LABEL_ANOMALY])}"
    )

    _save_confusion_matrix(
        cm_comb, "test_combined_all_cloned",
        "Confusion Matrix (Test Set) -- Real vs All Cloned"
    )
    _save_decision_function_plot(df_vals_real_test, per_source_df_vals)

    if clf is not None and hasattr(clf, "train_losses_"):
        _save_training_loss_plot(clf.train_losses_, getattr(clf, "val_losses_", None))
    elif clf is not None and hasattr(clf, "members_"):
        _save_training_loss_plot_ensemble(clf.members_)

    results = {
        "split": {
            "test_size_real":   TEST_SIZE_REAL,
            "test_size_cloned": TEST_SIZE_CLONED,
            "calibration_size": CALIBRATION_SIZE,
            "n_real_trainval":  int(len(X_real_tv)),
            "n_real_test":      int(len(X_real_test)),
            "cloned_trainval":  {n: int(len(v)) for n, v in cloned_tv.items()},
            "cloned_test":      {n: int(len(v)) for n, v in cloned_test.items()},
        },
        "final_model": _classifier_info(clf) if clf is not None else None,
        "test_real_audio": {
            "n_samples":           int(len(X_real_test)),
            "inlier_rate":         inlier_rate_test,
            "false_positive_rate": fpr_test,
            "n_predicted_inlier":  int(np.sum(y_pred_real_test == 1)),
            "n_predicted_outlier": int(np.sum(y_pred_real_test == -1)),
            "mean_decision_score": float(np.mean(df_vals_real_test)),
        },
        "per_source": per_source_results,
        "combined_all_cloned": {
            "sources":                list(cloned_test.keys()),
            "n_cloned_total":         int(len(X_all_cloned_test)),
            "anomaly_detection_rate": detection_rate_all,
            "false_negative_rate":    fnr_all,
            "accuracy":               float(acc_comb),
            "roc_auc":                float(auc_comb),
            "confusion_matrix":       cm_comb.tolist(),
            "classification_report":  rep_comb,
        },
    }

    return pipe, results


# ==============================================================================
# 11. SALVATAGGIO PLOT
# ==============================================================================

def _save_confusion_matrix(cm: np.ndarray, split_name: str, title: str):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=CONFUSION_MATRIX_CMAP,
        xticklabels=[LABEL_NORMAL, LABEL_ANOMALY],
        yticklabels=[LABEL_NORMAL, LABEL_ANOMALY],
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    out_path = PLOTS_DIR / f"confusion_matrix_{split_name}.png"
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info(f"  Confusion matrix salvata: {out_path}")


def _save_decision_function_plot(df_real_vals: np.ndarray, cloned_vals: dict):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(df_real_vals, bins=40, alpha=0.60, label=LABEL_NORMAL, color="steelblue", density=True)
    for i, (name, vals) in enumerate(cloned_vals.items()):
        color = CLONED_SOURCE_COLORS[i % len(CLONED_SOURCE_COLORS)]
        ax.hist(vals, bins=40, alpha=0.55, label=f"{LABEL_ANOMALY} ({name})", color=color, density=True)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="Soglia (0)")
    ax.set_xlabel("Decision function value  (threshold - MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Decision Function Distribution (Test Set)\nOne-Class Autoencoder -- Real vs Cloned",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out_path = PLOTS_DIR / "decision_function_distribution_test.png"
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info(f"  Decision function plot salvato: {out_path}")


def _save_training_loss_plot(train_losses: list, val_losses: list = None):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Training MSE")
    if val_losses:
        ax.plot(epochs, val_losses, color="darkorange", linewidth=1.5, label="Validation MSE (early stopping)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training/Validation Loss -- One-Class Autoencoder (modello finale)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = PLOTS_DIR / "training_loss_final_model.png"
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info(f"  Training loss plot salvato: {out_path}")


def _save_training_loss_plot_ensemble(members: list):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(members):
        color = cmap(i % 10)
        epochs = range(1, len(m.train_losses_) + 1)
        ax.plot(epochs, m.train_losses_, color=color, linewidth=1.2, alpha=0.85, label=f"Train (membro {i + 1})")
        if getattr(m, "val_losses_", None):
            ax.plot(epochs, m.val_losses_, color=color, linewidth=1.0, alpha=0.5, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training/Validation Loss -- Ensemble One-Class Autoencoder", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = PLOTS_DIR / "training_loss_final_model_ensemble.png"
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    log.info(f"  Training loss plot (ensemble) salvato: {out_path}")


# ==============================================================================
# 12. SALVATAGGIO METRICHE E MODELLO
# ==============================================================================

def save_results(
    pipe,
    kfold_results: dict,
    eval_results: dict,
    feature_cols: list,
    feature_selector,
    grid_search_results: dict = None,
):
    model_path = MODEL_DIR / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    log.info(f"\nModello salvato: {model_path}")

    if feature_selector is not None:
        selector_path = MODEL_DIR / "feature_selector.pkl"
        with open(selector_path, "wb") as f:
            pickle.dump(feature_selector, f)
        log.info(f"Feature selector salvato: {selector_path}")

    clf = pipe.named_steps.get("classifier")
    effective_params = _classifier_info(clf) if clf is not None else {}

    metrics = {
        "config": {
            "model":                        "One-Class Autoencoder (v3: calibrazione soglia robusta + ensemble)",
            "training_data":                "real audio train/val split (al netto del calibration set)",
            "test_sources":                 list(CLONED_SOURCES.keys()),
            "test_size_real":               TEST_SIZE_REAL,
            "test_size_cloned":             TEST_SIZE_CLONED,
            "calibration_size":             CALIBRATION_SIZE,
            "threshold_calibration_method": THRESHOLD_CALIBRATION_METHOD,
            "threshold_target_fpr":         THRESHOLD_TARGET_FPR,
            "threshold_n_bootstrap":        THRESHOLD_N_BOOTSTRAP,
            "threshold_n_candidates":       THRESHOLD_N_CANDIDATES,
            "scaler_type":                  SCALER_TYPE,
            "n_folds":                      N_FOLDS,
            "scale_features":               SCALE_FEATURES,
            "use_variance_threshold":       USE_VARIANCE_THRESHOLD,
            "use_feature_selection":        USE_FEATURE_SELECTION,
            "n_features_select":            N_FEATURES_SELECT if USE_FEATURE_SELECTION else None,
            "use_pca":                      USE_PCA,
            "pca_n_components":             PCA_N_COMPONENTS if USE_PCA else None,
            "n_features":                   len(feature_cols),
            "ae_ensemble_size":             AE_ENSEMBLE_SIZE,
            "ae_ensemble_bootstrap":        AE_ENSEMBLE_BOOTSTRAP,
            "grid_search_use_auc_scoring":  GRID_SEARCH_USE_AUC_SCORING,
            "autoencoder":                  effective_params,
        },
        "kfold_paired_trainval": kfold_results,
        "evaluation_test_set":   eval_results,
        "grid_search": {
            "enabled":     GRID_SEARCH_ENABLED,
            "randomized":  GRID_SEARCH_RANDOMIZED,
            "cv_type":     "PairedOneClassKFold",
            "scoring":     GRID_SEARCH_SCORING,
            "n_iter":      GRID_SEARCH_N_ITER if GRID_SEARCH_RANDOMIZED else None,
            "best_params": grid_search_results["best_params"] if grid_search_results else None,
            "best_score":  grid_search_results["best_score"]  if grid_search_results else None,
        },
    }
    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metriche salvate: {metrics_path}")
    log.info(
        f"\n  ROC-AUC finale (test combinato real vs all cloned): "
        f"{eval_results['combined_all_cloned']['roc_auc']:.4f}"
    )
    log.info(
        f"  FPR finale (test set, reali): "
        f"{eval_results['test_real_audio']['false_positive_rate']:.4f}"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    sources_str = ", ".join(CLONED_SOURCES.keys())
    device_str  = "CUDA" if (AE_USE_GPU and torch.cuda.is_available()) else "CPU"

    log.info("=" * 60)
    log.info("Audio Clone Detection -- One-Class Autoencoder (Novelty Detection)")
    log.info("VERSIONE MIGLIORATA v3: calibrazione soglia robusta (griglia di percentili")
    log.info("+ bootstrap, con vincolo esplicito sul FPR) per ridurre il gap FPR")
    log.info("k-fold/test rispetto alla v2, ensemble (bagging) per il modello finale,")
    log.info("holdout early-stopping/soglia disgiunti (fix leakage).")
    log.info(f"Training: audio reali  |  Test sources: {sources_str}")
    log.info(f"Holdout test size: reali={TEST_SIZE_REAL:.0%} (train/val={1 - TEST_SIZE_REAL:.0%})  |  clonati={TEST_SIZE_CLONED:.0%} (train/val={1 - TEST_SIZE_CLONED:.0%})")
    log.info(f"Calibrazione soglia: metodo={THRESHOLD_CALIBRATION_METHOD}  |  target_fpr={THRESHOLD_TARGET_FPR:.2f}")
    log.info(f"Device: {device_str}")
    log.info("=" * 60)
    log.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

    df_real, cloned_dfs = load_datasets(CSV_ORIGINAL_PATH, CLONED_SOURCES)

    X_real_tv, X_real_test, cloned_tv, cloned_test, feature_cols, feature_selector = (
        preprocess_datasets(df_real, cloned_dfs)
    )

    X_real_fit, X_real_calib, cloned_fit, cloned_calib = split_calibration_set(X_real_tv, cloned_tv)

    grid_search_results = None
    best_pipeline = None
    if GRID_SEARCH_ENABLED:
        grid_search_results = run_grid_search(X_real_fit, cloned_fit)
        best_pipeline = grid_search_results["best_pipeline"]

    kfold_results = run_kfold(X_real_fit, cloned_fit, best_pipeline)

    pipe, eval_results = train_and_evaluate(
        X_real_fit, X_real_test, cloned_fit, cloned_test, best_pipeline,
        X_real_calib=X_real_calib, cloned_calib=cloned_calib,
    )

    save_results(
        pipe, kfold_results, eval_results, feature_cols, feature_selector,
        grid_search_results,
    )

    log.info(f"\n{'=' * 60}")
    log.info("Pipeline completata con successo.")
    log.info(f"Output salvati in: {OUTPUT_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()