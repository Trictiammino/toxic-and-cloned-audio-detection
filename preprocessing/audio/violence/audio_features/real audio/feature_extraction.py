import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
TSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "real_audio_dataset.tsv"
SAVE_PATH = PROJECT_ROOT / "datasets" / "violence" / "extracted_audio_features" / "audio" / "features_real_audio_dataset.csv"

SR = 16000
N_MFCC = 20
N_MELS = 40


# --- UTILS ---
def safe_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x)
    }


# --- FEATURE EXTRACTION ---
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    features = {}

    # =====================
    # 🔹 ENERGY
    # =====================
    rms = librosa.feature.rms(y=y)[0]
    for k, v in safe_stats(rms).items():
        features[f"rms_{k}"] = v

    # =====================
    # 🔹 ZERO CROSSING
    # =====================
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    for k, v in safe_stats(zcr).items():
        features[f"zcr_{k}"] = v

    # =====================
    # 🔹 SPECTRAL FEATURES
    # =====================
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]

    for name, feat in {
        "centroid": spec_centroid,
        "bandwidth": spec_bandwidth,
        "rolloff": spec_rolloff,
        "flatness": spec_flatness
    }.items():
        for k, v in safe_stats(feat).items():
            features[f"{name}_{k}"] = v

    # =====================
    # 🔹 MFCC + DELTA
    # =====================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(N_MFCC):
        for name, arr in {
            "mfcc": mfcc[i],
            "mfcc_delta": delta[i],
            "mfcc_delta2": delta2[i]
        }.items():
            stats = safe_stats(arr)
            for k, v in stats.items():
                features[f"{name}_{i}_{k}"] = v

    # =====================
    # 🔹 MEL BANDS (KEY FOR FREQUENCY EXPLAINABILITY)
    # =====================
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)

    for i in range(N_MELS):
        band = mel_db[i]
        features[f"mel_{i}_mean"] = np.mean(band)
        features[f"mel_{i}_std"] = np.std(band)

    # =====================
    # 🔹 PITCH (F0)
    # =====================
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=50,
        fmax=400
    )

    f0 = f0[~np.isnan(f0)]
    if len(f0) > 0:
        for k, v in safe_stats(f0).items():
            features[f"pitch_{k}"] = v
    else:
        for k in ["mean", "std", "min", "max", "skew", "kurtosis"]:
            features[f"pitch_{k}"] = 0.0

    return features


# --- MAIN ---
df = pd.read_csv(TSV_PATH, sep="\t")

# =========================
# 🔹 MODIFICA: CARICAMENTO FILE ESISTENTE
# =========================
if SAVE_PATH.exists():
    df_existing = pd.read_csv(SAVE_PATH)
    processed_ids = set(df_existing["sample_id"])
else:
    processed_ids = set()

rows = []

print(f"🚀 Estrazione feature su {len(df)} file...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        sample_id = row["id"]

        # =========================
        # 🔹 MODIFICA: SKIP GIÀ PROCESSATI
        # =========================
        if sample_id in processed_ids:
            print(f"⏭️ SKIP già processato: {sample_id}")
            continue

        audio_path = str(PROJECT_ROOT / row["path"].lstrip("/"))
        label = 1 if row["toxicity"] == "Yes" else 0

        feats = extract_features(audio_path)

        feats["sample_id"] = row["id"]
        feats["path"] = row["path"]
        feats["text"] = row["text"]
        feats["label"] = label

        rows.append(feats)


    except Exception as e:

        print(f"\n❌ Errore sample {idx}")

        print(f"ID: {sample_id}")

        print(f"PATH: {audio_path}")

        print(f"Errore: {e}\n")

        continue


df_new = pd.DataFrame(rows)

# =========================
# 🔹 MODIFICA: APPEND SICURO
# =========================
if SAVE_PATH.exists():
    df_old = pd.read_csv(SAVE_PATH)
    df_out = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_out = df_new

df_out.to_csv(SAVE_PATH, index=False)

print(f"✅ Salvato in: {SAVE_PATH}")