# WALLpy v12 - archived local notes

This file preserves the pre-publication Italian notes supplied with the local
application. The maintained installation, usage, security, and citation
documentation is now in [`README.md`](README.md).

---

# Original WALLpy v12 notes

Nuova versione di WALLpy che aggiunge la **segmentazione con SAM 2**
(Segment Anything Model 2, Meta AI) accanto ai metodi già esistenti.

## Contenuto della cartella

| File | Descrizione |
|---|---|
| `WALLpy_v12.py` | Applicazione principale (GUI) |
| `sam2_segmentation.py` | Backend SAM 2 (caricamento modello, mask generation, fusione maschere) |
| `requirements_v12.txt` | Dipendenze Python |
| `checkpoints/` | Cartella per i checkpoint SAM 2 (solo backend ufficiale) |

I file della v11 (`pred.py`, `lib/`, `output/`, modello DL) **non sono stati
modificati**: la v12 li usa direttamente dalla cartella padre `C:\WALLpy`.

## Novità della v12

- **Pulsante "🪄 SAM 2"** nella barra superiore: segmenta la muratura con
  SAM 2 in modalità *automatic mask generation*. Ogni maschera proposta da
  SAM 2 è un candidato mattone/pietra; le maschere vengono filtrate per area
  (si scartano lo sfondo/parete intera e il rumore) e la loro unione diventa
  la maschera binaria mattoni/malta.
- Il risultato entra nel **workflow standard di WALLpy**: filtri morfologici,
  raffinamento colore, gomma manuale, visualizzazione bordi, export DXF e PNG.
- **Doppio backend** con fallback automatico:
  1. pacchetto ufficiale `sam2` (facebookresearch) + checkpoint locale;
  2. pacchetto `ultralytics` (download automatico dei pesi).
- **Cache del modello**: il modello SAM 2 resta in memoria tra una
  segmentazione e l'altra (la prima esecuzione è la più lenta).
- **Percorsi corretti dalla sottocartella**: il pulsante "🤖 Segmentazione DL"
  continua a funzionare anche se lo script è in `WALLpy_v12\` (i percorsi di
  `pred.py` e `config.yaml` sono risolti rispetto al project root).

## Installazione di SAM 2

L'ambiente esistente (Python 3.11, PyTorch 2.6 + CUDA 11.8) è già adeguato.
Serve solo **uno** dei due backend:

### Opzione A – ultralytics (consigliata, più semplice)

```
pip install ultralytics
```

I pesi (`sam2.1_b.pt`, ~160 MB) vengono scaricati automaticamente alla prima
segmentazione. Nessun altro passaggio richiesto.

### Opzione B – pacchetto ufficiale Meta

```
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Poi scaricare il checkpoint nella cartella `WALLpy_v12\checkpoints\`:

| Modello | File | Download |
|---|---|---|
| tiny | `sam2.1_hiera_tiny.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt |
| small | `sam2.1_hiera_small.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt |
| **base_plus** (default) | `sam2.1_hiera_base_plus.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt |
| large | `sam2.1_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt |

Se entrambi i backend sono installati, viene usato quello ufficiale.

## Avvio

```
cd C:\WALLpy
python WALLpy_v12\WALLpy_v12.py
```

Workflow: **📂 Carica Immagine** → **🪄 SAM 2** (oppure 🧮 K-Means /
🤖 DL) → raffinamento con slider/gomma → **📊 Esporta DXF** o
**⬛ Maschera → Esporta PNG**.

## Parametri SAM 2 (costanti in testa a `WALLpy_v12.py`)

| Costante | Default | Significato |
|---|---|---|
| `SAM2_MODEL_SIZE` | `"base_plus"` | Dimensione modello: `tiny` / `small` / `base_plus` / `large` |
| `SAM2_POINTS_PER_SIDE` | `32` | Densità della griglia di prompt: più alto = più elementi trovati, più lento (solo backend ufficiale) |
| `SAM2_PRED_IOU_THRESH` | `0.8` | Qualità minima delle maschere (solo backend ufficiale) |
| `SAM2_STABILITY_THRESH` | `0.92` | Stabilità minima delle maschere (solo backend ufficiale) |
| `SAM2_MIN_AREA_RATIO` | `0.0005` | Area minima di una maschera (frazione dell'immagine) perché sia un mattone |
| `SAM2_MAX_AREA_RATIO` | `0.35` | Area massima: maschere più grandi sono considerate sfondo e scartate |
| `SAM2_MAX_SIDE` | `1600` | Lato massimo per l'inferenza: immagini più grandi vengono ridotte e la maschera riportata alla risoluzione originale |

Suggerimenti:
- **Pietre molto grandi** non riconosciute → aumentare `SAM2_MAX_AREA_RATIO`.
- **Conci piccoli mancanti** → aumentare `SAM2_POINTS_PER_SIDE` (es. 48/64)
  e/o ridurre `SAM2_MIN_AREA_RATIO`.
- **Lentezza / GPU out of memory** → usare `"small"` o `"tiny"`, oppure
  ridurre `SAM2_MAX_SIDE`.
- I giunti di malta erroneamente inclusi si rimuovono con il filtro
  **Open** e con la **Soglia** di raffinamento colore, come nelle versioni
  precedenti.

## Note

- La prima esecuzione carica il modello (10–60 s a seconda del backend);
  le successive riusano il modello in cache.
- Con GPU (CUDA) l'inferenza su un'immagine ~1600 px richiede indicativamente
  10–60 s; su CPU diversi minuti.
- Se nessun backend è installato, il pulsante SAM 2 mostra un messaggio con
  le istruzioni di installazione (l'app resta pienamente utilizzabile con
  K-Means e DL).
