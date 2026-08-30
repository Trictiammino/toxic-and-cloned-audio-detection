import whisper
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "yt_audio.tsv"

# Carica modello
model = whisper.load_model("medium")

# Leggi TSV
df = pd.read_csv(TSV_PATH, sep="\t")

# Assicurati che esista la colonna text
if "text" not in df.columns:
    df["text"] = ""

for i, row in tqdm(df.iterrows(), total=len(df), desc="Trascrizione"):

    # Skip se già trascritto
    if pd.notna(row["text"]) and row["text"].strip() != "":
        continue

    # Path assoluto (corregge / iniziale)
    audio_file = PROJECT_ROOT / row["path"].lstrip("/")

    if audio_file.exists():
        try:
            result = model.transcribe(str(audio_file))
            df.loc[i, "text"] = result["text"].strip()
        except Exception as e:
            print(f"[ERROR] {audio_file}: {e}")
            df.loc[i, "text"] = ""
    else:
        print(f"[WARN] File non trovato: {audio_file}")
        df.loc[i, "text"] = ""

    # Salvataggio ogni 10 file
    if i % 10 == 0:
        df.to_csv(TSV_PATH, sep="\t", index=False)

# Rimuovi righe senza testo
df = df[df["text"].str.strip() != ""]

# Salva finale
df.to_csv(TSV_PATH, sep="\t", index=False)

print("Trascrizione completata!")