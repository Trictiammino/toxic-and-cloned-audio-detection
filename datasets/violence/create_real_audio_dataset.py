import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TSV1 = PROJECT_ROOT / "datasets" / "violence" / "filtered_audio.tsv"
TSV2  = PROJECT_ROOT / "datasets" / "violence" / "filtered_yt_audio.tsv"

OUTPUT_PATH = PROJECT_ROOT / "datasets" / "violence" / "real_audio_dataset.tsv"

# --- Caricamento ---
df_1 = pd.read_csv(TSV1, sep="\t")
df_2  = pd.read_csv(TSV2, sep="\t")

# --- (opzionale ma consigliato) uniforma colonne ---
columns = ["id", "path", "text", "toxicity"]
df_1_uniformed = df_1[columns]
df_2_uniformed  = df_2[columns]

# --- Merge ---
df_merged = pd.concat([df_1_uniformed, df_2_uniformed], ignore_index=True)

# --- SORT PER ID ---
df_merged = df_merged.sort_values(by="id").reset_index(drop=True)

# --- Salvataggio ---
df_merged.to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"✅ Dataset unito salvato in: {OUTPUT_PATH}")