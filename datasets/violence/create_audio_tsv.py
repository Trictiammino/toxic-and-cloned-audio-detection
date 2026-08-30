import os
import pandas as pd

audio_folder = "audio"
output_tsv = "audio.tsv"

base_path = "/datasets/violence/audio"

rows = []

for file_name in os.listdir(audio_folder):

    if not file_name.endswith(".wav"):
        continue

    file_id = os.path.splitext(file_name)[0]

    # 🔥 label basata sul nome file
    lower_name = file_name.lower()

    if lower_name.startswith("angry"):
        toxicity = "Yes"
    elif lower_name.startswith("noviolence"):
        toxicity = "No"
    else:
        # se non matcha skippo
        print(f"Skipping (nome non valido): {file_name}")
        continue

    path = f"{base_path}/{file_name}"

    rows.append([file_id, path, toxicity, ""])  # text vuoto

df = pd.DataFrame(rows, columns=["id", "path", "toxicity", "text"])

df.to_csv(output_tsv, sep="\t", index=False)

print(f"Creato manifest: {output_tsv} con {len(df)} file")