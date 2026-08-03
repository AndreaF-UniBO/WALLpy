#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official Meta SAM 2 backend for PyWALL v13.

Segmenta una muratura con SAM 2 (Segment Anything Model 2, Meta AI) in
modalità "automatic mask generation": SAM 2 propone una maschera per ogni
oggetto/regione dell'immagine; le maschere vengono filtrate per area e la
loro unione diventa la maschera binaria dei mattoni (255 = mattone,
0 = malta), nello stesso formato interno usato da PyWALL.

Il backend richiede il pacchetto ufficiale ``facebookresearch/sam2`` e un
checkpoint Meta locale nella cartella ``checkpoints``.

Nessuna dipendenza pesante viene importata al caricamento del modulo:
torch e sam2 sono importati solo alla prima segmentazione. PyWALL non scarica
automaticamente pacchetti o pesi.
"""

from pathlib import Path
import time

import numpy as np
import cv2


class Sam2NotAvailableError(RuntimeError):
    """Sollevata quando nessun backend SAM 2 è utilizzabile."""
    pass


# ------------------------------------------------------------------
# Mappe modello -> file checkpoint / config
# ------------------------------------------------------------------
# Pacchetto ufficiale `sam2`: (nomi checkpoint accettati, config yaml)
_OFFICIAL_MODELS = {
    "tiny": (
        ["sam2.1_hiera_tiny.pt", "sam2_hiera_tiny.pt"],
        {"sam2.1": "configs/sam2.1/sam2.1_hiera_t.yaml",
         "sam2": "configs/sam2/sam2_hiera_t.yaml"},
    ),
    "small": (
        ["sam2.1_hiera_small.pt", "sam2_hiera_small.pt"],
        {"sam2.1": "configs/sam2.1/sam2.1_hiera_s.yaml",
         "sam2": "configs/sam2/sam2_hiera_s.yaml"},
    ),
    "base_plus": (
        ["sam2.1_hiera_base_plus.pt", "sam2_hiera_base_plus.pt"],
        {"sam2.1": "configs/sam2.1/sam2.1_hiera_b+.yaml",
         "sam2": "configs/sam2/sam2_hiera_b+.yaml"},
    ),
    "large": (
        ["sam2.1_hiera_large.pt", "sam2_hiera_large.pt"],
        {"sam2.1": "configs/sam2.1/sam2.1_hiera_l.yaml",
         "sam2": "configs/sam2/sam2_hiera_l.yaml"},
    ),
}

_DOWNLOAD_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Cache dei modelli già caricati: {chiave: oggetto}
_CACHE = {}


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_official_checkpoint(model_size, checkpoint_dirs):
    """Cerca il checkpoint ufficiale nelle cartelle indicate.

    Ritorna (path, config_yaml) oppure (None, None).
    """
    ckpt_names, cfg_map = _OFFICIAL_MODELS[model_size]
    for d in checkpoint_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for name in ckpt_names:
            p = d / name
            if p.is_file():
                cfg = cfg_map["sam2.1"] if name.startswith("sam2.1") else cfg_map["sam2"]
                return p, cfg
    return None, None


def _load_official_generator(model_size, checkpoint_dirs, device,
                             points_per_side, pred_iou_thresh,
                             stability_score_thresh):
    """Carica (con cache) il SAM2AutomaticMaskGenerator ufficiale."""
    ckpt, cfg = _find_official_checkpoint(model_size, checkpoint_dirs)
    if ckpt is None:
        raise FileNotFoundError("checkpoint SAM 2 non trovato")

    key = ("official", str(ckpt.resolve()), device, points_per_side,
           pred_iou_thresh, stability_score_thresh)
    if key in _CACHE:
        return _CACHE[key]

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_model = build_sam2(cfg, str(ckpt), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=0,
        min_mask_region_area=0,  # il filtro per area è fatto a valle
    )
    _CACHE[key] = generator
    return generator


def _fuse_masks(mask_list, h, w, min_area_ratio, max_area_ratio):
    """Unisce le maschere-candidato in un'unica maschera mattoni.

    mask_list: lista di array bool (h, w). Le maschere troppo grandi
    (sfondo/parete intera) e troppo piccole (rumore) vengono scartate.

    Ritorna (mask uint8 {0,255}, n_accettate).
    """
    total = float(h * w)
    brick = np.zeros((h, w), dtype=np.uint8)
    kept = 0
    for m in mask_list:
        area = int(m.sum())
        ratio = area / total
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        brick[m] = 255
        kept += 1
    return brick, kept


def segment_wall_sam2(image_rgb,
                      model_size="base_plus",
                      device=None,
                      points_per_side=32,
                      pred_iou_thresh=0.8,
                      stability_score_thresh=0.92,
                      min_area_ratio=0.0005,
                      max_area_ratio=0.35,
                      max_side=1600,
                      checkpoint_dirs=None,
                      progress_callback=None):
    """Segmenta una muratura con SAM 2.

    Args:
      image_rgb: np.ndarray (H, W, 3) uint8, RGB.
      model_size: 'tiny' | 'small' | 'base_plus' | 'large'.
      device: 'cuda' | 'cpu' | None (auto).
      points_per_side: densità della griglia di prompt.
      pred_iou_thresh / stability_score_thresh: soglie di qualità delle
        maschere.
      min_area_ratio / max_area_ratio: area relativa (0..1) minima/massima
        di una maschera perché sia considerata un mattone.
      max_side: se il lato massimo supera questo valore l'inferenza avviene
        su un'immagine ridotta e la maschera viene riportata alla dimensione
        originale (nearest neighbour).
      checkpoint_dirs: cartelle in cui cercare i checkpoint ufficiali.
      progress_callback: funzione (str) -> None per aggiornare la GUI.

    Returns:
      (mask, info): mask uint8 (H, W) con 255 = mattone e 0 = malta;
      info dict con backend, device, n. maschere, tempi.

    Raises:
      Sam2NotAvailableError se nessun backend è utilizzabile.
    """
    def notify(msg):
        if progress_callback:
            progress_callback(msg)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Attesa immagine RGB (H, W, 3), ricevuto shape {image_rgb.shape}")
    if model_size not in _OFFICIAL_MODELS:
        raise ValueError(f"model_size non valido: {model_size!r}. "
                         f"Valori ammessi: {sorted(_OFFICIAL_MODELS)}")

    if checkpoint_dirs is None:
        here = Path(__file__).resolve().parent
        checkpoint_dirs = [here / "checkpoints", here.parent / "checkpoints"]

    h_orig, w_orig = image_rgb.shape[:2]

    # Eventuale riduzione per l'inferenza
    img_infer = image_rgb
    scale = 1.0
    if max(h_orig, w_orig) > max_side:
        scale = max_side / max(h_orig, w_orig)
        new_w = max(1, int(round(w_orig * scale)))
        new_h = max(1, int(round(h_orig * scale)))
        notify(f"Ridimensionamento per inferenza: {w_orig}x{h_orig} -> {new_w}x{new_h}")
        img_infer = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h, w = img_infer.shape[:2]

    t0 = time.time()
    errors = []

    # ---------- Backend 1: pacchetto ufficiale `sam2` ----------
    mask_list = None
    backend = None
    n_total = 0
    try:
        import sam2  # noqa: F401 - verifica solo la disponibilità
        sam2_installed = True
    except ImportError as e:
        sam2_installed = False
        errors.append(f"- pacchetto 'sam2' non installato ({e})")

    if sam2_installed:
        try:
            dev = device or _get_device()
            notify(f"Caricamento SAM 2 ({model_size}, backend ufficiale, {dev})...")
            generator = _load_official_generator(
                model_size, checkpoint_dirs, dev,
                points_per_side, pred_iou_thresh, stability_score_thresh)
            notify("Generazione automatica maschere (può richiedere qualche minuto)...")
            records = generator.generate(img_infer)
            records.sort(key=lambda r: r["area"], reverse=True)
            mask_list = [r["segmentation"].astype(bool) for r in records]
            n_total = len(mask_list)
            backend = "Meta SAM 2"
            device = dev
        except FileNotFoundError:
            ckpt_names = _OFFICIAL_MODELS[model_size][0]
            dirs_str = "\n    ".join(str(Path(d)) for d in checkpoint_dirs)
            errors.append(
                f"- pacchetto 'sam2' installato ma checkpoint mancante.\n"
                f"  Scarica '{ckpt_names[0]}' da:\n"
                f"    {_DOWNLOAD_BASE}/{ckpt_names[0]}\n"
                f"  e copialo in una di queste cartelle:\n    {dirs_str}")
        except Exception as e:
            errors.append(f"- backend 'sam2' fallito: {e}")

    if mask_list is None:
        msg = ("Il backend ufficiale Meta SAM 2 non è disponibile.\n\n"
               "Problemi riscontrati:\n" + "\n".join(errors) +
               "\n\nInstalla il pacchetto ufficiale da:\n"
               "  https://github.com/facebookresearch/sam2\n"
               "e colloca il checkpoint Meta richiesto nella cartella "
               "'checkpoints'.\n\n"
               "Dettagli nel file README_PyWALL_v13.md.")
        raise Sam2NotAvailableError(msg)

    t_infer = time.time() - t0

    # ---------- Fusione maschere -> maschera mattoni ----------
    notify("Fusione maschere...")
    brick_mask, n_kept = _fuse_masks(mask_list, h, w, min_area_ratio, max_area_ratio)

    # Riporta alla dimensione originale se era stata ridotta
    if scale != 1.0:
        brick_mask = cv2.resize(brick_mask, (w_orig, h_orig),
                                interpolation=cv2.INTER_NEAREST)

    info = {
        "backend": backend,
        "device": device,
        "model_size": model_size,
        "masks_total": n_total,
        "masks_kept": n_kept,
        "inference_scale": scale,
        "inference_time_s": round(t_infer, 2),
        "total_time_s": round(time.time() - t0, 2),
    }
    return brick_mask, info


def clear_cache():
    """Svuota la cache dei modelli (libera memoria GPU)."""
    _CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
