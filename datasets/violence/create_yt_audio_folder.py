from pathlib import Path
from pydub import AudioSegment

# ==========================================
# CONFIG
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_DIR = PROJECT_ROOT / "datasets" / "violence" / "yt_archive"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "violence" / "yt_audio"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# PARAMETRI
# ==========================================

FRAME_MS = 500
SILENCE_THRESHOLD = -40
MIN_SPEECH_MS = 1000

MERGE_GAP_MS = 500

CHUNK_MS = 10 * 1000

# elimina chunk troppo piccoli
MIN_CHUNK_MS = 7000

# ==========================================
# FILE WAV
# ==========================================

wav_files = sorted(INPUT_DIR.glob("*.wav"))

print(f"📃 Trovati {len(wav_files)} file WAV\n")

# ==========================================
# PROCESSING
# ==========================================

for wav_path in wav_files:

    base_name = wav_path.stem

    print("\n====================================")
    print(f"🎵 Processing: {base_name}")
    print("====================================")

    try:

        # ==========================================
        # LOAD AUDIO
        # ==========================================

        audio = AudioSegment.from_wav(wav_path)

        print(f"⏱️ Durata: {len(audio)/1000:.2f} sec")

        speech_segments = []
        current_segment = None

        # ==========================================
        # DETECT AUDIO SEGMENTS
        # ==========================================

        print("🔍 Analisi segmenti audio...")

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
            print("⚠️ Nessun segmento trovato")
            continue

        print(f"✅ Segmenti trovati: {len(speech_segments)}")

        # ==========================================
        # MERGE SEGMENTI VICINI
        # ==========================================

        merged = []

        for seg in speech_segments:

            if not merged:
                merged.append(list(seg))

            else:

                prev = merged[-1]

                if seg[0] - prev[1] < MERGE_GAP_MS:
                    prev[1] = seg[1]

                else:
                    merged.append(list(seg))

        print(f"🔗 Segmenti uniti: {len(merged)}")

        # ==========================================
        # CUT CHUNKS
        # ==========================================

        chunk_counter = 0

        print("✂️ Creazione chunk...")

        for start, end in merged:

            segment_audio = audio[start:end]

            for i in range(0, len(segment_audio), CHUNK_MS):

                chunk = segment_audio[i:i + CHUNK_MS]

                if len(chunk) < MIN_CHUNK_MS:
                    continue

                chunk_counter += 1

                chunk_name = f"{base_name}-{chunk_counter}.wav"

                chunk_path = OUTPUT_DIR / chunk_name

                chunk.export(chunk_path, format="wav")

                print(f"✅ Salvato: {chunk_name}")

        print(f"🏁 Totale chunk: {chunk_counter}")

    except Exception as e:

        print(f"❌ ERRORE con {base_name}")
        print(e)

print("\n🎉 PROCESSING COMPLETATO")