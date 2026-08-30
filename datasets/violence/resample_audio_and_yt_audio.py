from pathlib import Path
import librosa
import soundfile as sf

# =========================
# CONFIG
# =========================
FOLDERS = [
    Path("audio"),
    Path("yt_audio")
]

TARGET_SR = 24000

# =========================
# CONVERSIONE
# =========================
for folder in FOLDERS:

    files = list(folder.glob("*.wav"))

    print(f"\n📂 Cartella: {folder}")
    print(f"🎵 File trovati: {len(files)}\n")

    for i, file in enumerate(files, 1):

        try:
            # load + resample + mono
            audio, sr = librosa.load(
                file,
                sr=TARGET_SR,
                mono=True
            )

            # sovrascrive stesso file
            sf.write(
                file,
                audio,
                TARGET_SR,
                subtype="PCM_16"
            )

            print(
                f"[{i}/{len(files)}] "
                f"{file.name} -> OK "
                f"(24000 Hz, MONO, PCM_16)"
            )

        except Exception as e:
            print(f"[{i}/{len(files)}] {file.name} -> ERROR: {e}")

print("\n✅ Conversione completata")