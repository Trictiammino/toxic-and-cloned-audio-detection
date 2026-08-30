import os
import pandas as pd
from pydub import AudioSegment

input_folder = "archive/audios_VSD"
output_folder = "audio"
tsv_file = "archive/VSD.tsv"

os.makedirs(output_folder, exist_ok=True)

# parametri
FRAME_MS = 500          # dimensione finestra (0.5 sec)
SILENCE_THRESHOLD = -40 # dBFS (più alto = più selettivo)
MIN_SPEECH_MS = 1000    # minimo per considerare parlato
CHUNK_MS = 10 * 1000    # 10 secondi

# leggi TSV (solo per lista file unica)
df = pd.read_csv(tsv_file, sep="\t")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Global_file_name"])

unique_files = df["Global_file_name"].dropna().unique()

for base_name in unique_files:

    base_name = str(base_name).strip()
    input_path = os.path.join(input_folder, base_name + ".wav")

    if not os.path.exists(input_path):
        print(f"File non trovato: {input_path}")
        continue

    try:
        audio = AudioSegment.from_wav(input_path)

        speech_segments = []
        current_segment = None

        # 🔥 analisi frame per frame
        for i in range(0, len(audio), FRAME_MS):
            frame = audio[i:i + FRAME_MS]

            if frame.dBFS > SILENCE_THRESHOLD:
                if current_segment is None:
                    current_segment = [i, i + FRAME_MS]
                else:
                    current_segment[1] = i + FRAME_MS
            else:
                if current_segment is not None:
                    duration = current_segment[1] - current_segment[0]
                    if duration >= MIN_SPEECH_MS:
                        speech_segments.append(tuple(current_segment))
                    current_segment = None

        # ultimo segmento
        if current_segment is not None:
            duration = current_segment[1] - current_segment[0]
            if duration >= MIN_SPEECH_MS:
                speech_segments.append(tuple(current_segment))

        if not speech_segments:
            print(f"Nessun parlato in {base_name}")
            continue

        # 🔥 unisci segmenti vicini
        merged = []
        for seg in speech_segments:
            if not merged:
                merged.append(list(seg))
            else:
                prev = merged[-1]
                if seg[0] - prev[1] < 500:  # gap < 0.5 sec
                    prev[1] = seg[1]
                else:
                    merged.append(list(seg))

        counter = 0

        # 🔥 slicing in chunk da 10 secondi
        for start, end in merged:
            segment_audio = audio[start:end]

            for i in range(0, len(segment_audio), CHUNK_MS):
                chunk = segment_audio[i:i + CHUNK_MS]

                if len(chunk) < 1000:
                    continue

                counter += 1
                output_name = f"{base_name}-{counter}.wav"
                output_path = os.path.join(output_folder, output_name)

                chunk.export(output_path, format="wav")

                print(f"Salvato: {output_name}")

    except Exception as e:
        print(f"Errore con {base_name}: {e}")