import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import contextlib
from tqdm import tqdm
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# DEFINITION OF PATHS
TSV_PATH = PROJECT_ROOT / "datasets" / "violence" / "real_audio_dataset.tsv"
OUTPUT_DIR = PROJECT_ROOT / "generate_cloned_audio" / "chatterbox_tts" / "violence" / "cloned audio"
OUTPUT_TSV = PROJECT_ROOT / "generate_cloned_audio" / "chatterbox_tts" / "violence" / "cloned_audio_dataset.tsv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# crea file TSV con header se non esiste
if not OUTPUT_TSV.exists():
    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("id\tpath\ttext\ttoxicity\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# LOADING MODEL
model = ChatterboxTTS.from_pretrained(
    device=device
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

# SEPARAZIONE ID AUDIO TOSSICI E NON TOSSICI
df_yes = df_all[df_all["toxicity"].astype(str).str.strip() == "Yes"].reset_index(drop=True)
df_no = df_all[df_all["toxicity"].astype(str).str.strip() != "Yes"].reset_index(drop=True)

# CREA DISTRIBUZIONI UNIFORMI PER IL VALORE DELL'EXAGGERATION
yes_values = np.linspace(0.7, 1.5, len(df_yes))
no_values = np.linspace(0.2, 0.6, len(df_no))

# SHUFFLE
np.random.seed(42)
np.random.shuffle(yes_values)
np.random.shuffle(no_values)

 # MAPPING AUDIO ID CON VALORE EXAGGERATION
yes_map = dict(zip(df_yes["id"].astype(str), yes_values))
no_map = dict(zip(df_no["id"].astype(str), no_values))

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
    toxicity = str(row.toxicity).strip()

    if toxicity == "Yes":
        exaggeration_value = yes_map.get(audio_id)
    else:
        exaggeration_value = no_map.get(audio_id)

    # CLONED AUDIO GENERATION
    try:

        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull):
                with contextlib.redirect_stderr(fnull):

                    wav = model.generate(
                        text=ref_text,
                        audio_prompt_path=str(ref_audio),
                        exaggeration=exaggeration_value
                    )

        torch.cuda.synchronize()

        # converti in formato compatibile
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()

        wav = wav.squeeze().astype("float32")

        sf.write(str(output_path), wav, model.sr)

        del wav
        torch.cuda.empty_cache()

        if audio_id + "_cloned" in existing_ids:
            print(f"[SKIP] Già nel TSV: {audio_id}_cloned")
            continue

        relative_path = f"/generate_cloned_audio/chatterbox_tts/violence/cloned audio/{audio_id}_cloned.wav"

        toxicity = row.toxicity if "toxicity" in row else "Unknown"

        with open(OUTPUT_TSV, "a", encoding="utf-8") as f:
            f.write(f"{audio_id}_cloned\t{relative_path}\t{ref_text}\t{toxicity}\n")

        existing_ids.add(audio_id + "_cloned")

        print(f"\n[OK] Salvato: {audio_id}_cloned\n")

    except Exception as e:
        print(f"[ERROR] {audio_id}: {e}")