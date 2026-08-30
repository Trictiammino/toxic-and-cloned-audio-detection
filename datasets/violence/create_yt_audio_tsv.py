from pathlib import Path
import pandas as pd

# ==========================================
# CONFIG
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

AUDIO_FOLDER = PROJECT_ROOT / "datasets" / "violence" / "yt_audio"
OUTPUT_TSV = PROJECT_ROOT / "datasets" / "violence" / "yt_audio.tsv"

BASE_PATH = "/datasets/violence/yt_audio"

# ==========================================
# BUILD MANIFEST
# ==========================================

rows = []

wav_files = sorted(AUDIO_FOLDER.glob("*.wav"))

print(f"📃 Trovati {len(wav_files)} file WAV\n")

for wav_path in wav_files:

    file_name = wav_path.name

    file_id = wav_path.stem

    path = f"{BASE_PATH}/{file_name}"

    rows.append({
        "id": file_id,
        "path": path,
        "toxicity": "Yes",
        "text": ""
    })

    print(f"✅ Added: {file_name}")

# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame(rows)

# shuffle opzionale
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# SAVE TSV
# ==========================================

df.to_csv(OUTPUT_TSV, sep="\t", index=False)

print("\n================================")
print(f"🎉 TSV creato:")
print(f"{OUTPUT_TSV}")
print(f"📦 Totale samples: {len(df)}")
print("================================")