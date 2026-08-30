import yt_dlp
from pathlib import Path

# ==========================================
# CONFIGURAZIONE
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PLAYLIST_URL = "https://youtube.com/playlist?list=PLfzefYi-aMJ-83NWtNKYB4tzqURniRGof"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "violence" / "yt_archive"
OUTPUT_DIR.mkdir(exist_ok=True)

START_NUMBER = 22

# ==========================================
# CONTATORE GLOBALE
# ==========================================

current_number = START_NUMBER

# ==========================================
# HOOK PROGRESS BAR
# ==========================================

def progress_hook(d):
    global current_number

    if d["status"] == "downloading":
        percent = d.get("_percent_str", "").strip()
        speed = d.get("_speed_str", "N/A")
        eta = d.get("_eta_str", "N/A")

        print(
            f"\r⬇️  Downloading angry_{current_number} | "
            f"{percent} | Speed: {speed} | ETA: {eta}",
            end=""
        )

    elif d["status"] == "finished":
        print(f"\n🎵 Download completato per angry_{current_number}")

# ==========================================
# OTTIENI VIDEO DELLA PLAYLIST
# ==========================================

extract_opts = {
    "quiet": True,
    "extract_flat": True,
    "skip_download": True,
}

with yt_dlp.YoutubeDL(extract_opts) as ydl:
    info = ydl.extract_info(PLAYLIST_URL, download=False)

videos = info["entries"]

print(f"📃 Trovati {len(videos)} video nella playlist\n")

# ==========================================
# DOWNLOAD VIDEO UNO ALLA VOLTA
# ==========================================

for idx, video in enumerate(videos):

    video_url = f"https://www.youtube.com/watch?v={video['id']}"

    output_name = OUTPUT_DIR / f"angry_{current_number}.%(ext)s"

    print(f"\n==============================")
    print(f"🎬 Video {idx + 1}/{len(videos)}")
    print(f"🔗 {video_url}")
    print(f"💾 Salvataggio come angry_{current_number}.wav")
    print(f"==============================")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_name),

        # conversione WAV massima qualità
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],

        "progress_hooks": [progress_hook],

        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        wav_file = OUTPUT_DIR / f"angry_{current_number}.wav"

        if wav_file.exists():
            print(f"✅ SUCCESSO -> {wav_file.name}")
        else:
            print(f"❌ ERRORE -> angry_{current_number}.wav non trovato")

    except Exception as e:
        print(f"\n❌ FALLITO angry_{current_number}")
        print(f"Errore: {e}")

    current_number += 1

print("\n🏁 Download completati.")