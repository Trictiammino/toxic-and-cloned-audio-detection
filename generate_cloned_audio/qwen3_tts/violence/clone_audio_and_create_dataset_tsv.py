import os
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from qwen_tts import Qwen3TTSModel
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# DEFINITION OF PATHS
TSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "real_audio_dataset.tsv"
OUTPUT_DIR = PROJECT_ROOT / "generate_cloned_audio" / "qwen3_tts" / "violence" / "cloned audio"
OUTPUT_TSV = PROJECT_ROOT / "generate_cloned_audio" / "qwen3_tts" / "violence" / "cloned_audio_dataset.tsv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# crea file TSV con header se non esiste
if not OUTPUT_TSV.exists():
    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("id\tpath\ttext\ttoxicity\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# LOADING MODEL
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)


# DATASET TOXICITY (df_all = df.reset_index(drop=True).head(3) to try)
df = pd.read_csv(TSV_PATH, sep="\t")
df_all = df.reset_index(drop=True)


# COUNTER FOR TQDM
total = len(df_all)

# Output TSV ids set
existing_ids = set()

# Output TSV ids
if OUTPUT_TSV.exists():
    df_existing = pd.read_csv(OUTPUT_TSV, sep="\t")
    existing_ids = set(df_existing["id"].astype(str))

# LOOP CLONING
for idx, row in tqdm(df_all.iterrows(), total=total, desc="Cloning audio"):

    audio_id = str(row.id)
    ref_text = row.text
    ref_audio = PROJECT_ROOT / str(row.path).lstrip("/")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] audio number {idx+1}: {audio_id}")

    # OUTPUT PATH
    output_path = OUTPUT_DIR / f"{audio_id}_cloned.wav"

    # SKIP IF EMPTY TEXT
    if pd.isna(ref_text):
        print(f"[SKIP] text mancante: {audio_id}")
        continue

    # SKIP IF ALREADY EXISTS
    if output_path.exists():
        print(f"[SKIP] File già esistente: {output_path}\n")
        continue


    # AUDIO NOT FOUND
    if not ref_audio.exists():
        print(f"[SKIP] File non trovato: {ref_audio}\n")
        continue


    # CLONED AUDIO GENERATION
    try:
        wavs, sr = model.generate_voice_clone(
            text=ref_text,
            language="English",
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )

        torch.cuda.synchronize()

        # SAVING CLONED AUDIO
        sf.write(output_path, wavs[0], sr)

        del wavs
        torch.cuda.empty_cache()

        # Skip audio already in output TSV
        if audio_id+"_cloned" in existing_ids:
            print(f"[SKIP] Già nel TSV: {audio_id}_cloned")
            continue

        # path relativo
        relative_path = f"/generate_cloned_audio/qwen3_tts_voice_clone/violence/cloned audio/{audio_id}_cloned.wav"

        # toxicity
        toxicity = row.toxicity if "toxicity" in row else "Unknown"

        # append al TSV
        with open(OUTPUT_TSV, "a", encoding="utf-8") as f:
            f.write(f"{audio_id}_cloned\t{relative_path}\t{ref_text}\t{toxicity}\n")

        existing_ids.add(audio_id + "_cloned")

        print(f"\n[OK] Salvato: {audio_id}_cloned\n")




    except Exception as e:
        print(f"[ERROR] {audio_id}: {e}")
