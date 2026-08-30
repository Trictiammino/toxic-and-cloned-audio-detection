# Generazione e Rilevamento di Audio Clonati e Audio Tossici Tramite Feature Acustiche e Modelli Spiegabili

Codice e pipeline sperimentale della tesi di laurea in Sicurezza Informatica. Il progetto affronta due task di classificazione binaria condotti sullo stesso impianto sperimentale: la **rilevazione di voci clonate** (voice clone detection) e la **rilevazione di contenuti tossici nel parlato** (toxic audio detection), con l'obiettivo di valutare se la clonazione vocale possa sostituire i dati reali nella costruzione di dataset audio rispettosi della riservatezza.

|||
|-|-|
|**Autore**|Giuseppe Fantone|
|**Relatore**|Prof. Donato Impedovo|
|**Co-relatore**|Dott. Vincenzo Gattulli|
|**Corso di Laurea**|Sicurezza Informatica|
|**Ateneo**|Università degli Studi di Bari "Aldo Moro"|
|**Anno Accademico**|2025/2026|

## Indice

* [Introduzione](#introduzione)
* [Obiettivi](#obiettivi)
* [Costruzione del dataset audio, trascrizione ed estrazione delle feature](#costruzione-del-dataset-audio-trascrizione-ed-estrazione-delle-feature)

  * [Dataset reale: fonti e pipeline](#dataset-reale-fonti-e-pipeline)
  * [Generazione degli audio clonati](#generazione-degli-audio-clonati)
  * [Estrazione delle feature acustiche](#estrazione-delle-feature-acustiche)
* [Esperimenti e risultati](#esperimenti-e-risultati)

  * [Impianto sperimentale](#impianto-sperimentale)
  * [Rilevazione delle voci clonate (Voice Clone Detection)](#rilevazione-delle-voci-clonate-voice-clone-detection)
  * [Rilevazione dei contenuti tossici (Toxic Audio Detection)](#rilevazione-dei-contenuti-tossici-toxic-audio-detection)
  * [Spiegabilità (SHAP)](#spiegabilità-shap)
* [Struttura del progetto](#struttura-del-progetto)

  * [Albero delle cartelle](#albero-delle-cartelle)
  * [Guida rapida: dove trovare cosa](#guida-rapida-dove-trovare-cosa)
  * [Ordine di esecuzione della pipeline](#ordine-di-esecuzione-della-pipeline)
  * [Note pratiche](#note-pratiche)

## Introduzione

La diffusione dei canali di comunicazione vocale (chiamate, chat audio, streaming) ha esteso il problema del linguaggio tossico online dal testo scritto al parlato, rendendo necessari strumenti di rilevazione automatica capaci di operare direttamente sul segnale acustico, indipendentemente da una trascrizione testuale. In parallelo, i moderni sistemi di **voice cloning** neurale (sintesi vocale zero-shot) pongono un duplice interrogativo: da un lato rappresentano una sfida per l'autenticità dei contenuti audio, dall'altro offrono un'opportunità metodologica, ossia generare dati sintetici per l'addestramento di classificatori senza dover raccogliere, per ragioni legali ed etiche, registrazioni reali di persone coinvolte in situazioni di conflitto.

Questo progetto indaga entrambi gli aspetti attraverso lo stesso impianto sperimentale: a partire da un dataset reale di parlato tossico/non tossico in lingua inglese vengono generati due dataset sintetici tramite clonazione vocale (**Qwen3-TTS** e **ChatterboxTTS**), e sui tre corpora risultanti (reale + 2 clonati) vengono addestrati e confrontati gli stessi classificatori, misurando quanto la capacità di rilevare la tossicità sopravviva al processo di clonazione.

La scelta metodologica di fondo è l'uso di **feature acustiche interpretabili** (MFCC, descrittori spettrali, energetici e di pitch) invece di rappresentazioni apprese end-to-end, in modo da mantenere il processo di classificazione trasparente e analizzabile con tecniche di spiegabilità (SHAP).

## Obiettivi

Il lavoro si propone di rispondere a tre domande di ricerca:

1. **Rilevabilità dei cloni** — è possibile distinguere in modo affidabile un audio reale da uno clonato a partire da semplici feature acustiche, e quanto questa capacità dipende dal sistema di sintesi impiegato (Qwen3-TTS vs ChatterboxTTS)?
2. **Sopravvivenza dei marcatori di tossicità** — i classificatori acustici riescono a rilevare la tossicità verbale sia su audio reali sia su audio clonati, e quali descrittori acustici risultano maggiormente coinvolti nella decisione?
3. **Generalizzazione one-class** — un autoencoder addestrato su una sola classe (solo audio reali o solo audio clonati) è in grado di riconoscere le voci clonate generalizzando anche a sistemi di sintesi non visti in addestramento, a differenza dei classificatori supervisionati vincolati alle sole architetture osservate in training?

L'obiettivo finale è trarre indicazioni sulla concreta praticabilità della clonazione vocale come strategia per costruire dataset audio destinati alla rilevazione di contenuti tossici, bilanciando utilità scientifica e tutela della riservatezza dei soggetti coinvolti.

## Costruzione del dataset audio, trascrizione ed estrazione delle feature

### Dataset reale: fonti e pipeline

Il dataset reale è costruito a partire da due fonti (script in `datasets/violence/`):

* **Violence Sound Dataset (VSD)** — dataset pubblico di audio violenti; 342 segmenti ricavati da 21 file audio di circa 10 minuti, con manifest originale in formato TSV/XLSX contenente gli intervalli temporali di violenza.
* **YouTube** — 18 video scaricati da una playlist pubblica tramite `yt-dlp`, da cui sono stati ricavati 277 segmenti aggiuntivi di litigi e interazioni verbalmente aggressive, etichettati interamente come tossici.

Pipeline di costruzione:

1. **Segmentazione VSD** (`create_audio_folder.py`) — rilevamento del parlato tramite analisi frame-per-frame dell'energia (finestra 500 ms, soglia -40 dBFS, parlato minimo 1000 ms), unione dei segmenti con gap < 500 ms, suddivisione in chunk da 10 s; etichettatura in base al nome del file originale (`angry\*` → tossico, `noviolence\*` → non tossico) e creazione del manifest con `create_audio_tsv.py`.
2. **Download e segmentazione YouTube** (`create_yt_archive_folder.py`, `create_yt_audio_folder.py`, `create_yt_audio_tsv.py`) — stessa logica di rilevamento/segmentazione applicata ai video scaricati, tutti etichettati come tossici.
3. **Resampling comune** (`resample_audio_and_yt_audio.py`) — tutti gli audio vengono uniformati a 24 kHz, mono, 16 bit PCM.
4. **Trascrizione** (`audio_transcription.py`, `yt_audio_transcription.py`) — trascrizione automatica con **Whisper medium** (OpenAI); i campioni privi di contenuto verbale intelligibile vengono scartati.
5. **Fusione** (`create_real_audio_dataset.py`) — le due sorgenti filtrate vengono unite e ordinate per id in `real_audio_dataset.tsv` (colonne `id`, `path`, `text`, `toxicity`).

Il dataset reale finale conta **2.394 campioni** (1.604 non tossici, 790 tossici — rapporto di circa 2:1).

> Nota: i manifest TSV contengono le trascrizioni testuali degli audio, che includono linguaggio volgare, offensivo o violento, in quanto necessario a rappresentare fedelmente il fenomeno studiato.

### Generazione degli audio clonati

Per ciascun campione del dataset reale viene generata una replica sintetica tramite due sistemi text-to-speech con voice cloning **zero-shot** (audio originale come riferimento timbrico + relativa trascrizione come testo da pronunciare), script in `generate_cloned_audio/`:

* **Qwen3-TTS** (`qwen3_tts/violence/clone_audio_and_create_dataset_tsv.py`) — variante `Qwen3-TTS-12Hz-1.7B-Base` in precisione bfloat16; clonazione neutra ad alta fedeltà, senza controllo esplicito dell'espressività.
* **ChatterboxTTS** (`chatterbox_tts/violence/clone_audio_and_create_dataset_tsv.py`) — modello open source di Resemble AI (backbone Llama da circa 0,5 miliardi di parametri, architettura Flow Matching non autoregressiva); il parametro di **esagerazione emotiva** è modulato in base all'etichetta di tossicità (0,7–1,5 per i campioni tossici, 0,2–0,6 per i non tossici, distribuzione uniforme con seed fisso 42).

Ogni script produce i file `.wav` clonati e un manifest `cloned_audio_dataset.tsv` con lo stesso schema del dataset reale (id con suffisso `_cloned`). Risultato: 2.393 campioni per Chatterbox (1 fallimento di sintesi) e 2.394 per Qwen3, con la stessa distribuzione tossico/non tossico del dataset reale.

### Estrazione delle feature acustiche

Da ciascun campione di ognuno dei tre dataset (reale, cloni Chatterbox, cloni Qwen3) viene estratto un vettore di **482 feature acustiche interpretabili** tramite `librosa` (audio caricato mono a 16 kHz), aggregando media, deviazione standard, minimo, massimo, asimmetria e curtosi delle serie temporali dei descrittori. Script `feature_extraction.py` in `preprocessing/audio/violence/audio_features/`:

|Gruppo di feature|Descrizione|N. feature|
|-|-|-|
|Energia (RMS)|Radice quadratica media dell'energia di trama|6|
|Zero-Crossing Rate|Tasso di attraversamento dello zero|6|
|Spettrali|Centroide, bandwidth, roll-off, flatness|24|
|MFCC + Δ + ΔΔ|20 coefficienti cepstrali e relative derivate prima/seconda|360|
|Bande Mel|40 bande del melspettrogramma in scala dB (solo media/std)|80|
|Pitch (F0)|Frequenza fondamentale stimata con pYIN (50–400 Hz)|6|
|**Totale**||**482**|

Le feature estratte vengono salvate in formato CSV (`features_real_audio_dataset.csv` / `features_cloned_audio_dataset.csv`) con colonne aggiuntive `sample_id`, `path`, `text`, `label` (0 = non tossico, 1 = tossico).

## Esperimenti e risultati

### Impianto sperimentale

Entrambi i task condividono la stessa metodologia di validazione: split stratificato 85% training / 15% test (seed fisso 42), ottimizzazione degli iperparametri con `RandomizedSearchCV` (30 combinazioni campionate, 3-fold interno), validazione tramite 5-fold stratified cross-validation sul training set, valutazione finale sul test set held-out. La metrica di ottimizzazione è **ROC-AUC** per il voice clone detection (classi bilanciate per costruzione) e **F1-macro** per il toxic audio detection (classi sbilanciate \~2:1, gestite con `class_weight="balanced"` per Logistic Regression/Random Forest e `scale_pos_weight` per XGBoost).

Il piano sperimentale comprende **17 esperimenti** complessivi: 6 di voice clone detection supervisionato (3 classificatori × 2 scenari), 2 di voice clone detection one-class (autoencoder × 2 configurazioni di addestramento) e 9 di toxic audio detection (3 classificatori × 3 condizioni audio).

### Rilevazione delle voci clonate (Voice Clone Detection)

**Classificazione binaria supervisionata** — Logistic Regression, Random Forest e XGBoost addestrati a distinguere audio reale (`No`) da audio clonato (`Yes`), separatamente per Chatterbox e Qwen3:

|Classificatore|Scenario|Accuracy (test)|ROC-AUC (test)|F1-score (test)|
|-|-|-|-|-|
|**Logistic Regression**|**real vs Chatterbox**|**98,46%**|**99,88%**|**98,46%**|
|**Logistic Regression**|**real vs Qwen3**|**92,73%**|**97,09%**|**92,73%**|
|Random Forest|real vs Chatterbox|93,85%|98,87%|93,85%|
|Random Forest|real vs Qwen3|89,66%|96,58%|89,66%|
|XGBoost|real vs Chatterbox|97,07%|99,68%|97,07%|
|XGBoost|real vs Qwen3|92,74%|98,30%|92,73%|

I cloni **Qwen3** risultano sistematicamente più difficili da rilevare rispetto a quelli **Chatterbox**, a indicare una maggiore fedeltà acustica al parlato naturale.

**Classificazione one-class (autoencoder)** — un autoencoder undercomplete (ensemble bagging di 5 modelli, feature selection a 40 dimensioni via mutual information) viene addestrato su una sola classe alla volta e classifica come anomalo ciò che si discosta dalla distribuzione appresa, senza richiedere di osservare in training il sistema di sintesi da rilevare:

|Addestrato su|Classe anomala (test)|Accuracy|ROC-AUC|Detection rate|
|-|-|-|-|-|
|Audio reali|Cloni (Qwen3 + Chatterbox)|68,37%|80,16%|66,42%|
|Audio clonati (Qwen3 + Chatterbox)|Audio reali|68,04%|64,50%|49,79%|

Le prestazioni sono inferiori rispetto ai classificatori supervisionati, ma confermano la fattibilità di un riconoscimento dei cloni potenzialmente generalizzabile a sistemi di sintesi non visti in addestramento.

### Rilevazione dei contenuti tossici (Toxic Audio Detection)

I tre classificatori vengono addestrati e valutati separatamente su audio reali, cloni Chatterbox e cloni Qwen3:

|Classificatore|Condizione|Accuracy (test)|ROC-AUC (test)|Recall classe tossica|
|-|-|-|-|-|
|Logistic Regression|Audio reali|99,44%|99,98%|99,15%|
|Logistic Regression|Cloni Chatterbox|98,33%|99,30%|95,80%|
|Logistic Regression|Cloni Qwen3|94,99%|98,99%|88,13%|
|Random Forest|Audio reali|98,33%|99,74%|98,31%|
|Random Forest|Cloni Chatterbox|97,49%|99,82%|93,30%|
|Random Forest|Cloni Qwen3|94,15%|98,81%|83,90%|
|**XGBoost**|**Audio reali**|**99,44%**|**99,81%**|**99,16%**|
|**XGBoost**|**Cloni Chatterbox**|**99,16%**|**99,93%**|**97,48%**|
|**XGBoost**|**Cloni Qwen3**|**95,82%**|**99,35%**|**88,98%**|

XGBoost si conferma il modello più robusto in tutte le condizioni. Il recall sulla classe tossica, la metrica più critica in questo contesto, poiché una mancata rilevazione costituisce un errore più grave di un falso positivo, cala sensibilmente sui cloni Qwen3, a indicare che una maggiore fedeltà acustica del sistema di sintesi non garantisce un'equivalente conservazione dei marcatori paralinguistici della tossicità.

### Spiegabilità (SHAP)

Per ogni esperimento di toxic audio detection viene calcolata l'importanza delle feature con **SHAP** (`LinearExplainer` per Logistic Regression, `TreeExplainer` per Random Forest e XGBoost), producendo per ciascun campione le feature più rilevanti e, a livello globale, un beeswarm plot. La feature con importanza SHAP più alta varia sistematicamente in base al tipo di audio analizzato:

|Classificatore|Audio reali|Cloni Chatterbox|Cloni Qwen3|
|-|-|-|-|
|Logistic Regression|`mfcc_14_mean`|`mfcc_delta2_19_std`|`mfcc_1_max`|
|Random Forest|`mfcc_delta2_19_min`|`pitch_mean`|`mfcc_delta_18_min`|
|XGBoost|`mfcc_delta2_1_std`|`pitch_mean`|`pitch_mean`|

Nei cloni, in particolare per Chatterbox, la **frequenza fondamentale media (`pitch_mean`)** domina la classificazione; negli audio reali il potere discriminativo si distribuisce invece su feature di variabilità spettrale a bassa frequenza (delta dei MFCC, bandwidth).

## Struttura del progetto

### Albero delle cartelle

```text
Fantone - tesi/
├── datasets/
│   └── violence/
│       ├── archive/                          # sorgente originale VSD
│       │   ├── VSD.xlsx, VSD.tsv              # manifest con gli intervalli di violenza
│       │   ├── audios_VSD/                    # audio integrali VSD (\~10 min l'uno)
│       │   └── xlsx_to_tsv.py
│       ├── audio/                             # chunk da 10s segmentati dal VSD
│       ├── yt_archive/                        # video scaricati dalla playlist YouTube
│       ├── yt_audio/                          # chunk da 10s segmentati dagli audio YouTube
│       ├── extracted_audio_features/
│       │   └── audio/
│       │       └── features_real_audio_dataset.csv   # 482 feature del dataset reale
│       ├── audio.tsv, yt_audio.tsv                    # manifest post-segmentazione
│       ├── filtered_audio.tsv, filtered_yt_audio.tsv  # manifest post-trascrizione/filtraggio
│       ├── real_audio_dataset.tsv                     # dataset reale finale (2.394 campioni)
│       ├── create_audio_folder.py, create_audio_tsv.py            # segmentazione + manifest VSD
│       ├── create_yt_archive_folder.py, create_yt_audio_folder.py,
│       │   create_yt_audio_tsv.py                                 # download + segmentazione YouTube
│       ├── resample_audio_and_yt_audio.py                         # resampling 24kHz / mono / 16 bit
│       ├── audio_transcription.py, yt_audio_transcription.py      # trascrizione Whisper medium
│       └── create_real_audio_dataset.py                           # fusione delle due sorgenti
│
├── generate_cloned_audio/
│   ├── qwen3_tts/violence/
│   │   ├── cloned audio/                          # .wav generati da Qwen3-TTS
│   │   ├── extracted_audio_features/audio/
│   │   │   └── features_cloned_audio_dataset.csv  # 482 feature dei cloni Qwen3
│   │   ├── cloned_audio_dataset.tsv               # manifest dei cloni Qwen3
│   │   └── clone_audio_and_create_dataset_tsv.py
│   └── chatterbox_tts/violence/
│       ├── cloned audio/                          # .wav generati da ChatterboxTTS
│       ├── extracted_audio_features/audio/
│       │   └── features_cloned_audio_dataset.csv  # 482 feature dei cloni Chatterbox
│       ├── cloned_audio_dataset.tsv               # manifest dei cloni Chatterbox
│       └── clone_audio_and_create_dataset_tsv.py
│
├── preprocessing/audio/violence/audio_features/
│   ├── real audio/
│   │   └── feature_extraction.py           # -> datasets/.../features_real_audio_dataset.csv
│   └── cloned audio/
│       ├── qwen3_tts/feature_extraction.py       # -> generate_cloned_audio/qwen3_tts/...
│       └── chatterbox_tts/feature_extraction.py  # -> generate_cloned_audio/chatterbox_tts/...
│
└── classifiers/
    ├── logistic regression/violence/
    │   ├── voice_clone_detection/
    │   │   ├── real_vs_chatterbox/
    │   │   │   ├── train.py
    │   │   │   ├── model/      -> classifier.pkl
    │   │   │   ├── metrics/    -> metrics.json
    │   │   │   └── plots/      -> confusion_matrix_test.png
    │   │   └── real_vs_qwen3/             (stessa struttura)
    │   └── toxic_audio_detection/
    │       ├── real audio/
    │       │   ├── train.py
    │       │   ├── model/          -> pipeline.pkl, best_audio_classifier.pkl, best_audio_scaler.pkl
    │       │   ├── metrics/        -> metrics.json
    │       │   ├── plots/          -> confusion_matrix_test.png, shap_beeswarm.png
    │       │   └── top_features/   -> shap_values.npy, test_set_top_features.json
    │       ├── cloned audio/chatterbox_tts/   (stessa struttura)
    │       └── cloned audio/qwen3_tts/        (stessa struttura)
    │
    ├── random forest/violence/             # stessa struttura di "logistic regression/"
    ├── xgboost/violence/                   # stessa struttura di "logistic regression/"
    │
    └── one_class_autoencoder/violence/voice_clone_detection/
        ├── train_real.py                   # fit sui reali, valutazione sui cloni come anomalia
        ├── train_cloned.py                 # fit sui cloni, valutazione sui reali come anomalia
        ├── train_real_results/{model, metrics, plots}
        └── train_cloned_results/{model, metrics, plots}
```

### Guida rapida: dove trovare cosa

|Cosa cercare|Percorso|
|-|-|
|Dataset reale (manifest)|`datasets/violence/real_audio_dataset.tsv`|
|Audio clonati con Qwen3-TTS|`generate_cloned_audio/qwen3_tts/violence/cloned audio/`|
|Audio clonati con ChatterboxTTS|`generate_cloned_audio/chatterbox_tts/violence/cloned audio/`|
|Feature acustiche estratte (482-dim, CSV)|cartelle `extracted_audio_features/audio/` (una per ciascuna delle 3 sorgenti audio)|
|Modelli addestrati (`.pkl`)|`classifiers/<classificatore>/violence/<task>/.../model/`|
|Metriche di valutazione (`metrics.json`)|`classifiers/<classificatore>/violence/<task>/.../metrics/`|
|Grafici (confusion matrix, beeswarm SHAP, loss)|`classifiers/<classificatore>/violence/<task>/.../plots/`|
|Spiegabilità SHAP per singolo campione|`classifiers/<classificatore>/violence/toxic_audio_detection/.../top_features/`|
|Script di training|file `train.py` (o `train_real.py`/`train_cloned.py` per l'autoencoder) in ciascuna sottocartella di `classifiers/`|

### Ordine di esecuzione della pipeline

Gli script sono pensati per essere eseguiti nell'ordine seguente, poiché ogni fase legge gli output della precedente:

1. `datasets/violence/` — costruzione del dataset reale (segmentazione → resampling → trascrizione → fusione) fino a `real_audio_dataset.tsv`.
2. `generate_cloned_audio/qwen3_tts/` e `generate_cloned_audio/chatterbox_tts/` — generazione dei due dataset sintetici a partire da `real_audio_dataset.tsv`.
3. `preprocessing/audio/violence/audio_features/` — estrazione delle 482 feature acustiche per ciascuna delle tre sorgenti (reale, Qwen3, Chatterbox).
4. `classifiers/` — addestramento e valutazione di Logistic Regression, Random Forest, XGBoost (per entrambi i task) e dell'autoencoder one-class (solo per voice clone detection).

### Note pratiche

* **Percorsi assoluti dal root del progetto.** Gli script calcolano `PROJECT_ROOT` risalendo la propria posizione nell'albero delle cartelle e leggono/scrivono percorsi relativi ad esso: vanno quindi lanciati mantenendo intatta la struttura delle cartelle del repository.
* **Librerie principali.** `librosa` e `soundfile`/`pydub` per l'audio, `openai-whisper` per la trascrizione, `yt-dlp` per il download da YouTube, `chatterbox-tts` e `qwen-tts` per la clonazione vocale, `scikit-learn` e `xgboost` per i classificatori, `torch` per l'autoencoder, `shap` per la spiegabilità, `pandas`/`numpy` per la gestione dei dati.

