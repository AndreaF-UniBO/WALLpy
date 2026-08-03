#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyWALL v13: desktop tools for masonry-image segmentation.

The public v13 application provides two independent workflows:

* K-Means segmentation, available with the core installation;
* automatic mask generation with Meta's official SAM 2 backend.

SAM 2 model weights are loaded from the local ``checkpoints`` directory and
are never downloaded implicitly by the application.
"""

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import generic_filter
import warnings
import time
import traceback
from numpy.linalg import norm

# Modulo locale: backend SAM 2 (import leggero, torch caricato solo all'uso)
import sam2_segmentation

__version__ = "0.13.0"

# ================================================
#  OPTIONAL: ttkbootstrap (fallback a ttk se assente)
# ================================================
USE_TTKB = False
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    USE_TTKB = True
except Exception:
    import tkinter.ttk as ttk

try:
    import ezdxf
except ImportError:
    print("ATTENZIONE: La libreria 'ezdxf' non è installata.")
    print("Per usare l'esportazione DXF, installala con: pip install ezdxf")
    ezdxf = None

# --- Percorsi ---
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Costanti ---
TARGET_BRICK_COLOR_RGB = (255, 100, 0)
TARGET_MORTAR_COLOR_RGB = (255, 255, 255)
N_CLUSTERS = 3
MAX_DISPLAY_WIDTH = 400
MAX_DISPLAY_HEIGHT = 400
TEXTURE_WINDOW_SIZE = 7
EDGE_LOW_THRESHOLD = 50
EDGE_HIGH_THRESHOLD = 150
INITIAL_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 21
INITIAL_THRESHOLD = 0.80
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.5
INITIAL_CLEAN_KERNEL_SIZE = 1
MAX_CLEAN_KERNEL_SIZE = 11
ERASER_BRUSH_SIZE = 9
BRICK_MASK_VALUE = 255
MORTAR_MASK_VALUE = 0
CONTOUR_COLOR_BGR = (0, 255, 0)
DEFAULT_SCALE_FACTOR = 0.0005  # m/px (0.5 mm)
BACKGROUND_COLOR_GRAY = 128

# --- Costanti SAM 2 ---
SAM2_MODEL_SIZE = "base_plus"      # tiny | small | base_plus | large
SAM2_POINTS_PER_SIDE = 32          # densità griglia prompt (backend ufficiale)
SAM2_PRED_IOU_THRESH = 0.8         # soglia qualità maschere
SAM2_STABILITY_THRESH = 0.92       # soglia stabilità maschere
SAM2_MIN_AREA_RATIO = 0.0005       # area minima maschera (frazione immagine)
SAM2_MAX_AREA_RATIO = 0.35         # area massima (scarta sfondo/parete intera)
SAM2_MAX_SIDE = 1600               # lato max per inferenza (poi upscale mask)


# --- Funzioni di Elaborazione (preservate dall'originale) ---
def process_image_initial_segmentation(image_path, k=N_CLUSTERS):
    try:
        print("Caricamento e preparazione immagine...")
        start_time = time.time()
        original_img_pil = Image.open(image_path).convert('RGB')
        original_img_rgb = np.array(original_img_pil)
        h, w, _ = original_img_rgb.shape

        print("Estrazione caratteristiche...")
        img_lab = rgb2lab(original_img_rgb)
        img_gray = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, EDGE_LOW_THRESHOLD, EDGE_HIGH_THRESHOLD)
        edges_normalized = (edges / 255.0).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            local_std = generic_filter(img_gray.astype(np.float32), np.std, size=TEXTURE_WINDOW_SIZE)

        min_std, max_std = np.min(local_std), np.max(local_std)
        texture_normalized = (local_std - min_std) / (max_std - min_std) if max_std > min_std else np.zeros_like(local_std)

        features = np.zeros((h * w, 5), dtype=np.float32)
        features[:, 0:3] = img_lab.reshape(-1, 3)
        features[:, 3] = edges_normalized.reshape(-1) * 100
        features[:, 4] = texture_normalized.reshape(-1) * 100

        print(f"Applicazione K-Means con K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        avg_texture_per_cluster = np.array([np.mean(features[labels == i, 4]) for i in range(k)])
        mortar_cluster_index = np.argmin(avg_texture_per_cluster)

        print(f"Cluster identificato come malta: {mortar_cluster_index}")

        labels_2d = labels.reshape(h, w)
        pixels_lab = img_lab.reshape(-1, 3)
        mortar_pixels_lab = pixels_lab[labels == mortar_cluster_index]
        brick_pixels_lab = pixels_lab[labels != mortar_cluster_index]

        avg_mortar_lab = np.mean(mortar_pixels_lab, axis=0) if len(mortar_pixels_lab) > 0 else np.array([80.0, 0.0, 0.0])
        avg_brick_lab = np.mean(brick_pixels_lab, axis=0) if len(brick_pixels_lab) > 0 else np.array([50.0, 20.0, 20.0])

        end_time = time.time()
        print(f"Segmentazione completata in {end_time - start_time:.2f} secondi.")

        return original_img_pil, labels_2d, mortar_cluster_index, img_lab, avg_mortar_lab, avg_brick_lab
    except Exception as e:
        messagebox.showerror("Errore Segmentazione", f"Errore durante la segmentazione: {e}")
        return None, None, None, None, None, None

def apply_filters_refine_clean_and_recolor(base_mask, mortar_index, img_lab, avg_mortar_lab, avg_brick_lab,
                                           open_ksize, close_ksize, median_ksize, clean_ksize,
                                           brick_color_rgb=TARGET_BRICK_COLOR_RGB,
                                           mortar_color_rgb=TARGET_MORTAR_COLOR_RGB,
                                           refinement_threshold=INITIAL_THRESHOLD):
    if base_mask is None or img_lab is None:
        return None, None

    binary_mask = np.where(base_mask == mortar_index, MORTAR_MASK_VALUE, BRICK_MASK_VALUE).astype(np.uint8)
    filtered_mask = binary_mask.copy()

    if open_ksize > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel_open)

    if close_ksize > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_close)

    if median_ksize > 1:
        filtered_mask = cv2.medianBlur(filtered_mask, median_ksize)

    color_refined_mask = filtered_mask.copy()
    candidate_brick_mask = (filtered_mask == BRICK_MASK_VALUE)
    original_lab_bricks = img_lab[candidate_brick_mask]

    if len(original_lab_bricks) > 0:
        dist_to_mortar = norm(original_lab_bricks - avg_mortar_lab, axis=1)
        dist_to_brick = norm(original_lab_bricks - avg_brick_lab, axis=1)
        ratio = np.divide(dist_to_mortar, dist_to_brick, out=np.full_like(dist_to_mortar, np.inf), where=dist_to_brick > 1e-6)
        reclassify_as_mortar_indices = np.where(ratio < refinement_threshold)[0]
        coords_candidate_bricks = np.argwhere(candidate_brick_mask)
        coords_to_reclassify = coords_candidate_bricks[reclassify_as_mortar_indices]
        if len(coords_to_reclassify) > 0:
            color_refined_mask[coords_to_reclassify[:, 0], coords_to_reclassify[:, 1]] = MORTAR_MASK_VALUE

    cleaned_mask = color_refined_mask.copy()
    if clean_ksize > 1:
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_ksize, clean_ksize))
        cleaned_mask = cv2.morphologyEx(color_refined_mask, cv2.MORPH_OPEN, kernel_clean)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_clean)

    processed_img_pil = recolor_from_mask(cleaned_mask, img_lab, brick_color_rgb, mortar_color_rgb)
    return processed_img_pil, cleaned_mask

def recolor_from_mask(mask, img_lab_original, brick_color_rgb, mortar_color_rgb):
    if mask is None or img_lab_original is None:
        return None

    h, w = mask.shape
    processed_img_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    final_mortar_mask = (mask == MORTAR_MASK_VALUE)
    final_brick_mask = (mask == BRICK_MASK_VALUE)

    processed_img_rgb[final_mortar_mask] = mortar_color_rgb

    if np.any(final_brick_mask):
        brick_color_lab = rgb2lab(np.uint8([[brick_color_rgb]]))[0][0]
        lab_original_bricks_final = img_lab_original[final_brick_mask]
        if len(lab_original_bricks_final) > 0:
            lab_new_bricks = np.zeros_like(lab_original_bricks_final)
            lab_new_bricks[:, 0] = lab_original_bricks_final[:, 0]
            lab_new_bricks[:, 1] = brick_color_lab[1]
            lab_new_bricks[:, 2] = brick_color_lab[2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                rgb_new_bricks = lab2rgb(lab_new_bricks)
            processed_img_rgb[final_brick_mask] = np.clip(rgb_new_bricks * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(processed_img_rgb)

# --- Classe Applicazione ---
class PyWALLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyWALL v13 – Segmentazione murature")
        self.root.geometry("1400x800")

        if USE_TTKB:
            try:
                self.root.style.theme_use("darkly")
            except Exception:
                pass

        # Stato
        self.original_pil_image = None
        self.processed_pil_image = None
        self.processed_pil_with_contours = None
        self.processed_pil_contours_only = None
        self.image_path = None
        self.base_mask = None
        self.mortar_index = None
        self.img_lab = None
        self.avg_mortar_lab = None
        self.avg_brick_lab = None
        self.cleaned_mask_auto = None
        self.manual_mask = None
        self.brick_contours = None
        self.erasing_active = False
        self.eraser_brush_size = ERASER_BRUSH_SIZE
        self.show_contours_active = False
        self.show_background_with_contours = True
        self.processed_display_params = {'x_offset': 0, 'y_offset': 0, 'resized_w': 1, 'resized_h': 1, 'original_w': 1, 'original_h': 1}

        self.binary_mask_window = None
        self.binary_mask_window_photo = None
        self.pil_mask_for_display = None

        # Variabili controlli
        self.open_ksize_var = tk.IntVar(value=INITIAL_KERNEL_SIZE)
        self.close_ksize_var = tk.IntVar(value=INITIAL_KERNEL_SIZE)
        self.median_ksize_var = tk.IntVar(value=INITIAL_KERNEL_SIZE)
        self.threshold_var = tk.DoubleVar(value=INITIAL_THRESHOLD)
        self.clean_ksize_var = tk.IntVar(value=INITIAL_CLEAN_KERNEL_SIZE)
        self.scale_factor_var = tk.StringVar(value=str(DEFAULT_SCALE_FACTOR))

        # Build UI
        self._build_topbar()
        self._build_main_area()
        self._build_statusbar()

        # Photo refs
        self.original_photo = None
        self.processed_photo = None

        self._update_ui_state('initial')

    def _build_topbar(self):
        """Costruisce la barra superiore con azioni principali"""
        self.topbar = ttk.Frame(self.root, padding=6)
        self.topbar.pack(fill=tk.X)

        # Gruppo File
        file_frame = ttk.Frame(self.topbar)
        file_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.load_button = ttk.Button(file_frame, text="📂 Carica Immagine", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=2)

        self.process_button = ttk.Button(file_frame, text="🧮 K-Means", command=self.run_initial_segmentation, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=2)

        self.process_sam2_button = ttk.Button(file_frame, text="🪄 SAM 2", command=self.run_sam2_segmentation, state=tk.DISABLED)
        self.process_sam2_button.pack(side=tk.LEFT, padx=2)

        # Gruppo Strumenti
        tools_frame = ttk.Frame(self.topbar)
        tools_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.eraser_button = ttk.Button(tools_frame, text="✏️ Gomma", command=self._toggle_eraser, state=tk.DISABLED)
        self.eraser_button.pack(side=tk.LEFT, padx=2)

        self.reset_eraser_button = ttk.Button(tools_frame, text="↺ Reset", command=self._reset_manual_edits, state=tk.DISABLED)
        self.reset_eraser_button.pack(side=tk.LEFT, padx=2)

        # Gruppo Visualizzazione
        view_frame = ttk.Frame(self.topbar)
        view_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.show_contours_button = ttk.Button(view_frame, text="🔍 Bordi", command=self._toggle_show_contours, state=tk.DISABLED)
        self.show_contours_button.pack(side=tk.LEFT, padx=2)

        self.toggle_bg_button = ttk.Button(view_frame, text="🎨 Sfondo", command=self._toggle_background_visibility, state=tk.DISABLED)
        self.toggle_bg_button.pack(side=tk.LEFT, padx=2)

        self.show_binary_mask_button = ttk.Button(view_frame, text="⬛ Maschera", command=self._show_binary_mask_window, state=tk.DISABLED)
        self.show_binary_mask_button.pack(side=tk.LEFT, padx=2)

        # Scala
        scale_frame = ttk.Frame(self.topbar)
        scale_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(scale_frame, text="Scala (m/px):").pack(side=tk.LEFT, padx=(0, 5))
        self.scale_entry = ttk.Entry(scale_frame, textvariable=self.scale_factor_var, width=8)
        self.scale_entry.pack(side=tk.LEFT)

    def _build_main_area(self):
        """Costruisce l'area principale con canvas e controlli"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Canvas area (sinistra)
        self.canvas_area = ttk.Frame(self.main_container)
        self.canvas_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Original image frame
        orig_frame = ttk.LabelFrame(self.canvas_area, text="Immagine Originale", padding=8)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self.original_canvas = tk.Canvas(orig_frame, bg='#2b2b2b' if USE_TTKB else 'lightgrey', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Processed image frame
        proc_frame = ttk.LabelFrame(self.canvas_area, text="Immagine Elaborata", padding=8)
        proc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        self.processed_canvas = tk.Canvas(proc_frame, bg='#2b2b2b' if USE_TTKB else 'lightgrey',
                                         cursor="arrow", highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        # Controls panel (destra)
        self.controls_panel = ttk.Frame(self.main_container, width=380, padding=8)
        self.controls_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.controls_panel.pack_propagate(False)

        # Titolo controlli
        title_frame = ttk.Frame(self.controls_panel)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="Controlli Post-Processing", font=("", 12, "bold")).pack(anchor="w")

        # Card Filtri Morfologici
        morph_card = ttk.LabelFrame(self.controls_panel, text="Filtri Morfologici", padding=10)
        morph_card.pack(fill=tk.X, pady=(0, 10))

        # Open
        ttk.Label(morph_card, text="Open (Kernel):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.open_slider = ttk.Scale(morph_card, from_=1, to=MAX_KERNEL_SIZE, orient=tk.HORIZONTAL,
                                    variable=self.open_ksize_var, command=self._filter_slider_changed)
        self.open_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.open_label = ttk.Label(morph_card, text=f"{INITIAL_KERNEL_SIZE}", width=3)
        self.open_label.grid(row=0, column=2, pady=2)

        # Close
        ttk.Label(morph_card, text="Close (Kernel):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.close_slider = ttk.Scale(morph_card, from_=1, to=MAX_KERNEL_SIZE, orient=tk.HORIZONTAL,
                                     variable=self.close_ksize_var, command=self._filter_slider_changed)
        self.close_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.close_label = ttk.Label(morph_card, text=f"{INITIAL_KERNEL_SIZE}", width=3)
        self.close_label.grid(row=1, column=2, pady=2)

        # Median
        ttk.Label(morph_card, text="Median Blur:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.median_slider = ttk.Scale(morph_card, from_=1, to=MAX_KERNEL_SIZE, orient=tk.HORIZONTAL,
                                      variable=self.median_ksize_var, command=self._filter_slider_changed)
        self.median_slider.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.median_label = ttk.Label(morph_card, text=f"{INITIAL_KERNEL_SIZE}", width=3)
        self.median_label.grid(row=2, column=2, pady=2)

        morph_card.columnconfigure(1, weight=1)

        # Card Raffinamento Colore
        color_card = ttk.LabelFrame(self.controls_panel, text="Raffinamento Colore", padding=10)
        color_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(color_card, text="Soglia:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threshold_slider = ttk.Scale(color_card, from_=MIN_THRESHOLD, to=MAX_THRESHOLD, orient=tk.HORIZONTAL,
                                         variable=self.threshold_var, command=self._filter_slider_changed)
        self.threshold_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.threshold_label = ttk.Label(color_card, text=f"{INITIAL_THRESHOLD:.2f}", width=5)
        self.threshold_label.grid(row=0, column=2, pady=2)

        color_card.columnconfigure(1, weight=1)

        # Card Pulizia
        clean_card = ttk.LabelFrame(self.controls_panel, text="Pulizia Artefatti", padding=10)
        clean_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(clean_card, text="Clean Kernel:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.clean_slider = ttk.Scale(clean_card, from_=1, to=MAX_CLEAN_KERNEL_SIZE, orient=tk.HORIZONTAL,
                                     variable=self.clean_ksize_var, command=self._filter_slider_changed)
        self.clean_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.clean_label = ttk.Label(clean_card, text=f"{INITIAL_CLEAN_KERNEL_SIZE}", width=3)
        self.clean_label.grid(row=0, column=2, pady=2)

        clean_card.columnconfigure(1, weight=1)

        # Card Export
        export_card = ttk.LabelFrame(self.controls_panel, text="Esportazione", padding=10)
        export_card.pack(fill=tk.X, pady=(0, 10))

        self.export_dxf_button = ttk.Button(export_card, text="📊 Esporta DXF", command=self._export_dxf, state=tk.DISABLED)
        self.export_dxf_button.pack(fill=tk.X, pady=2)

        # Info area
        self.info_text = tk.Text(self.controls_panel, height=8, wrap=tk.WORD, bg='#2b2b2b' if USE_TTKB else 'white',
                                fg='white' if USE_TTKB else 'black', font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.info_text.config(state=tk.DISABLED)

        self._set_filter_controls_state(tk.DISABLED)

    def _build_statusbar(self):
        """Costruisce la barra di stato"""
        self.statusbar = ttk.Frame(self.root)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(self.statusbar, text="Pronto. Carica un'immagine per iniziare.", anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=4)

        self.progress_label = ttk.Label(self.statusbar, text="", anchor="e")
        self.progress_label.pack(side=tk.RIGHT, padx=10, pady=4)

    def _update_info(self, text):
        """Aggiorna il pannello informazioni"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)

    def _set_status(self, text, progress=None):
        """Aggiorna la barra di stato"""
        self.status_label.config(text=text)
        if progress:
            self.progress_label.config(text=progress)
        else:
            self.progress_label.config(text="")
        self.root.update_idletasks()

    def _update_ui_state(self, state):
        """Aggiorna lo stato dell'interfaccia"""
        if state == 'initial':
            self.process_button.config(state=tk.DISABLED)
            self.process_sam2_button.config(state=tk.DISABLED)
            self.eraser_button.config(state=tk.DISABLED)
            self.reset_eraser_button.config(state=tk.DISABLED)
            self.show_contours_button.config(state=tk.DISABLED)
            self.toggle_bg_button.config(state=tk.DISABLED)
            self.show_binary_mask_button.config(state=tk.DISABLED)
            self.export_dxf_button.config(state=tk.DISABLED)
            self._set_filter_controls_state(tk.DISABLED)
        elif state == 'image_loaded':
            self.process_button.config(state=tk.NORMAL)
            self.process_sam2_button.config(state=tk.NORMAL)
            self.eraser_button.config(state=tk.DISABLED)
            self.reset_eraser_button.config(state=tk.DISABLED)
            self.show_contours_button.config(state=tk.DISABLED)
            self.toggle_bg_button.config(state=tk.DISABLED)
            self.show_binary_mask_button.config(state=tk.DISABLED)
            self.export_dxf_button.config(state=tk.DISABLED)
            self._set_filter_controls_state(tk.DISABLED)
        elif state == 'segmented':
            self.process_button.config(state=tk.NORMAL)
            self.process_sam2_button.config(state=tk.NORMAL)
            self.eraser_button.config(state=tk.NORMAL)
            self.show_contours_button.config(state=tk.NORMAL)
            self.show_binary_mask_button.config(state=tk.NORMAL)
            self._set_filter_controls_state(tk.NORMAL)

    def _set_process_buttons_state(self, state):
        """Abilita/disabilita i pulsanti di caricamento e segmentazione"""
        self.load_button.config(state=state)
        self.process_button.config(state=state)
        self.process_sam2_button.config(state=state)

    def _resize_image_for_canvas(self, img_pil, canvas_w, canvas_h, resampling_method):
        if img_pil is None or canvas_w <= 1 or canvas_h <= 1:
            return None
        img_w, img_h = img_pil.size
        if img_w == 0 or img_h == 0:
            return None
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))
        try:
            return img_pil.resize((new_w, new_h), resampling_method)
        except Exception as e:
            print(f"Errore resize: {e}")
            return None

    def _set_filter_controls_state(self, state):
        widgets = [self.open_slider, self.close_slider, self.median_slider,
                  self.threshold_slider, self.clean_slider]
        for widget in widgets:
            widget.config(state=state)
        if state == tk.DISABLED:
            self.toggle_bg_button.config(state=tk.DISABLED)
        else:
            self.toggle_bg_button.config(state=tk.NORMAL if self.show_contours_active else tk.DISABLED)

    def _update_slider_label(self, label_widget, value, is_kernel=True, max_k=MAX_KERNEL_SIZE):
        if is_kernel:
            int_value = int(float(value))
            odd_value = int_value if int_value % 2 != 0 else int_value + 1
            odd_value = max(1, min(odd_value, max_k))
            label_widget.config(text=f"{odd_value}")
            return odd_value
        else:
            float_value = float(value)
            label_widget.config(text=f"{float_value:.2f}")
            return float_value

    def _reset_filter_controls(self):
        """Riporta slider e label dei filtri ai valori iniziali"""
        self.open_ksize_var.set(INITIAL_KERNEL_SIZE)
        self.close_ksize_var.set(INITIAL_KERNEL_SIZE)
        self.median_ksize_var.set(INITIAL_KERNEL_SIZE)
        self.threshold_var.set(INITIAL_THRESHOLD)
        self.clean_ksize_var.set(INITIAL_CLEAN_KERNEL_SIZE)

        self._update_slider_label(self.open_label, INITIAL_KERNEL_SIZE, max_k=MAX_KERNEL_SIZE)
        self._update_slider_label(self.close_label, INITIAL_KERNEL_SIZE, max_k=MAX_KERNEL_SIZE)
        self._update_slider_label(self.median_label, INITIAL_KERNEL_SIZE, max_k=MAX_KERNEL_SIZE)
        self._update_slider_label(self.threshold_label, INITIAL_THRESHOLD, is_kernel=False)
        self._update_slider_label(self.clean_label, INITIAL_CLEAN_KERNEL_SIZE, max_k=MAX_CLEAN_KERNEL_SIZE)

    def _filter_slider_changed(self, value):
        self._close_binary_mask_window_if_open()
        if self.erasing_active:
            self._toggle_eraser()
        if self.show_contours_active:
            self._toggle_show_contours(force_off=True)

        self.manual_mask = None
        self.brick_contours = None
        self.reset_eraser_button.config(state=tk.DISABLED)
        self.export_dxf_button.config(state=tk.DISABLED)

        open_k = self._update_slider_label(self.open_label, self.open_ksize_var.get(), max_k=MAX_KERNEL_SIZE)
        close_k = self._update_slider_label(self.close_label, self.close_ksize_var.get(), max_k=MAX_KERNEL_SIZE)
        median_k = self._update_slider_label(self.median_label, self.median_ksize_var.get(), max_k=MAX_KERNEL_SIZE)
        threshold = self._update_slider_label(self.threshold_label, self.threshold_var.get(), is_kernel=False)
        clean_k = self._update_slider_label(self.clean_label, self.clean_ksize_var.get(), max_k=MAX_CLEAN_KERNEL_SIZE)

        self.open_ksize_var.set(open_k)
        self.close_ksize_var.set(close_k)
        self.median_ksize_var.set(median_k)
        self.clean_ksize_var.set(clean_k)

        self.update_processed_image()
        self.root.update_idletasks()

    def display_image(self, canvas, img_pil, photo_ref_attr, is_processed=False):
        canvas.delete("all")
        if img_pil:
            container_w = canvas.winfo_width()
            container_h = canvas.winfo_height()

            if container_w <= 1 or container_h <= 1:
                canvas.update_idletasks()
                container_w = canvas.winfo_width()
                container_h = canvas.winfo_height()

            if container_w <= 1:
                container_w = MAX_DISPLAY_WIDTH
            if container_h <= 1:
                container_h = MAX_DISPLAY_HEIGHT

            img_resized = self._resize_image_for_canvas(img_pil, container_w, container_h, Image.Resampling.LANCZOS)
            if img_resized is None:
                setattr(self, photo_ref_attr, None)
                return

            try:
                photo = ImageTk.PhotoImage(img_resized)
                setattr(self, photo_ref_attr, photo)
                x_pos = max(0, (container_w - img_resized.width) // 2)
                y_pos = max(0, (container_h - img_resized.height) // 2)
                canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)

                if is_processed:
                    self.processed_display_params = {
                        'x_offset': x_pos, 'y_offset': y_pos,
                        'resized_w': img_resized.width, 'resized_h': img_resized.height,
                        'original_w': img_pil.width, 'original_h': img_pil.height
                    }
            except Exception as e:
                print(f"Errore display: {e}")
                setattr(self, photo_ref_attr, None)
        else:
            setattr(self, photo_ref_attr, None)
            if is_processed:
                self.processed_display_params = {
                    'x_offset': 0, 'y_offset': 0,
                    'resized_w': 1, 'resized_h': 1,
                    'original_w': 1, 'original_h': 1
                }

    def load_image(self):
        self._close_binary_mask_window_if_open()
        path = filedialog.askopenfilename(
            title="Seleziona Immagine",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        self.original_pil_image = None
        self.processed_pil_image = None
        self.processed_pil_with_contours = None
        self.processed_pil_contours_only = None
        self.base_mask = None
        self.mortar_index = None
        self.img_lab = None
        self.avg_mortar_lab = None
        self.avg_brick_lab = None
        self.cleaned_mask_auto = None
        self.manual_mask = None
        self.brick_contours = None
        self.erasing_active = False
        self.show_contours_active = False
        self.show_background_with_contours = True
        self.pil_mask_for_display = None

        self._update_ui_state('initial')
        self._set_status("Caricamento immagine...")

        try:
            self.original_pil_image = Image.open(self.image_path).convert('RGB')
            self.root.update()
            self.display_image(self.original_canvas, self.original_pil_image, "original_photo")
            self.display_image(self.processed_canvas, None, "processed_photo", is_processed=True)

            # Info immagine
            w, h = self.original_pil_image.size
            info = f"File: {os.path.basename(self.image_path)}\n"
            info += f"Dimensioni: {w} × {h} px\n"
            info += f"Formato: {self.original_pil_image.format or 'N/A'}\n"
            info += f"Modalità: {self.original_pil_image.mode}\n"
            self._update_info(info)

            self._update_ui_state('image_loaded')
            self._set_status(f"Immagine caricata: {os.path.basename(self.image_path)}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare l'immagine.\n{e}")
            self.image_path = None
            self.original_pil_image = None
            self._update_ui_state('initial')
            self._set_status("Errore caricamento immagine")

    def run_initial_segmentation(self):
        self._close_binary_mask_window_if_open()
        if not self.image_path or not self.original_pil_image:
            return

        self._set_status("Segmentazione in corso...", "K-Means clustering")
        self._set_process_buttons_state(tk.DISABLED)

        if self.erasing_active:
            self._toggle_eraser()
        if self.show_contours_active:
            self._toggle_show_contours(force_off=True)

        self.manual_mask = None
        self.brick_contours = None
        self.pil_mask_for_display = None

        _, self.base_mask, self.mortar_index, self.img_lab, self.avg_mortar_lab, self.avg_brick_lab = \
            process_image_initial_segmentation(self.image_path, k=N_CLUSTERS)

        self._set_process_buttons_state(tk.NORMAL)

        if all(v is not None for v in [self.base_mask, self.mortar_index, self.img_lab]):
            self._set_status("Segmentazione completata", "Applica i filtri per raffinare")

            self._reset_filter_controls()

            # Info segmentazione
            info = f"SEGMENTAZIONE COMPLETATA\n"
            info += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            info += f"Cluster malta: #{self.mortar_index}\n"
            info += f"Lab malta: L*={self.avg_mortar_lab[0]:.1f}, "
            info += f"a*={self.avg_mortar_lab[1]:.1f}, b*={self.avg_mortar_lab[2]:.1f}\n"
            info += f"Lab mattoni: L*={self.avg_brick_lab[0]:.1f}, "
            info += f"a*={self.avg_brick_lab[1]:.1f}, b*={self.avg_brick_lab[2]:.1f}\n\n"
            info += "Usa i controlli per raffinare\nil risultato."
            self._update_info(info)

            self.update_processed_image()
            self._update_ui_state('segmented')
        else:
            self._set_status("Errore segmentazione")
            self.display_image(self.processed_canvas, None, "processed_photo", True)

    def _adopt_binary_mask(self, internal_mask):
        """Integra una maschera binaria {0,255} (255=mattoni) nel workflow.

        Imposta base_mask/mortar_index/img_lab e i colori Lab medi, come
        richiesto da apply_filters_refine_clean_and_recolor.
        """
        self.base_mask = internal_mask
        self.mortar_index = 0
        self.img_lab = rgb2lab(np.array(self.original_pil_image))

        mortar_pixels = self.img_lab[internal_mask == MORTAR_MASK_VALUE]
        brick_pixels = self.img_lab[internal_mask == BRICK_MASK_VALUE]

        self.avg_mortar_lab = np.mean(mortar_pixels, axis=0) if len(mortar_pixels) > 0 else np.array([80.0, 0.0, 0.0])
        self.avg_brick_lab = np.mean(brick_pixels, axis=0) if len(brick_pixels) > 0 else np.array([50.0, 20.0, 20.0])

    def run_sam2_segmentation(self):
        """Esegue la segmentazione con SAM 2 (automatic mask generation).

        Ogni maschera proposta da SAM 2 è un candidato mattone/pietra; le
        maschere vengono filtrate per area e la loro unione diventa la
        maschera mattoni. Il risultato entra nel workflow standard
        (filtri, gomma, bordi, export DXF/PNG).
        """
        self._close_binary_mask_window_if_open()
        if not self.image_path or not self.original_pil_image:
            return

        self._set_status("Segmentazione SAM 2 in corso...", "Preparazione")
        self._set_process_buttons_state(tk.DISABLED)

        if self.erasing_active:
            self._toggle_eraser()
        if self.show_contours_active:
            self._toggle_show_contours(force_off=True)

        self.manual_mask = None
        self.brick_contours = None
        self.pil_mask_for_display = None

        try:
            image_rgb = np.array(self.original_pil_image)

            def progress(msg):
                self._set_status("Segmentazione SAM 2 in corso...", msg)

            internal_mask, sam_info = sam2_segmentation.segment_wall_sam2(
                image_rgb,
                model_size=SAM2_MODEL_SIZE,
                points_per_side=SAM2_POINTS_PER_SIDE,
                pred_iou_thresh=SAM2_PRED_IOU_THRESH,
                stability_score_thresh=SAM2_STABILITY_THRESH,
                min_area_ratio=SAM2_MIN_AREA_RATIO,
                max_area_ratio=SAM2_MAX_AREA_RATIO,
                max_side=SAM2_MAX_SIDE,
                checkpoint_dirs=[SCRIPT_DIR / "checkpoints"],
                progress_callback=progress,
            )

            if sam_info["masks_kept"] == 0:
                messagebox.showwarning(
                    "SAM 2",
                    "SAM 2 non ha individuato mattoni/pietre nell'immagine.\n"
                    f"(maschere proposte: {sam_info['masks_total']}, "
                    "tutte scartate dai filtri per area)\n\n"
                    "Prova ad aumentare SAM2_MAX_AREA_RATIO o a ridurre "
                    "SAM2_MIN_AREA_RATIO nelle costanti in testa al file.")

            # Setup per compatibilità con workflow esistente
            self._adopt_binary_mask(internal_mask)

            self._set_status("Segmentazione SAM 2 completata",
                             f"{sam_info['masks_kept']} elementi in {sam_info['total_time_s']}s")

            self._reset_filter_controls()

            # Info
            mortar_pct = (np.sum(internal_mask == MORTAR_MASK_VALUE) / internal_mask.size) * 100
            brick_pct = 100 - mortar_pct

            info = f"SEGMENTAZIONE SAM 2 COMPLETATA\n"
            info += f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            info += f"Backend: {sam_info['backend']}\n"
            info += f"Modello: {sam_info['model_size']} ({sam_info['device']})\n"
            info += f"Maschere: {sam_info['masks_kept']}/{sam_info['masks_total']} accettate\n"
            if sam_info['inference_scale'] != 1.0:
                info += f"Scala inferenza: {sam_info['inference_scale']:.2f}\n"
            info += f"Tempo: {sam_info['total_time_s']} s\n"
            info += f"Malta: {mortar_pct:.1f}%\n"
            info += f"Mattoni: {brick_pct:.1f}%\n\n"
            info += "Usa i controlli per raffinare\nil risultato."
            self._update_info(info)

            self.update_processed_image()
            self._update_ui_state('segmented')

        except sam2_segmentation.Sam2NotAvailableError as e:
            messagebox.showerror("SAM 2 non disponibile", str(e))
            self._set_status("SAM 2 non disponibile", "Vedi README_PyWALL_v13.md")
        except Exception as e:
            messagebox.showerror("Errore Segmentazione SAM 2", f"Errore:\n{str(e)}")
            self._set_status("Errore segmentazione SAM 2")
            traceback.print_exc()
        finally:
            self._set_process_buttons_state(tk.NORMAL)

    def update_processed_image(self):
        if not all(v is not None for v in [self.base_mask, self.mortar_index, self.img_lab]):
            self.show_binary_mask_button.config(state=tk.DISABLED)
            return

        open_k = self.open_ksize_var.get()
        close_k = self.close_ksize_var.get()
        median_k = self.median_ksize_var.get()
        threshold = self.threshold_var.get()
        clean_k = self.clean_ksize_var.get()

        temp_processed_pil, temp_cleaned_mask = apply_filters_refine_clean_and_recolor(
            self.base_mask, self.mortar_index, self.img_lab,
            self.avg_mortar_lab, self.avg_brick_lab,
            open_k, close_k, median_k, clean_k,
            TARGET_BRICK_COLOR_RGB, TARGET_MORTAR_COLOR_RGB, threshold
        )

        if temp_processed_pil is not None and temp_cleaned_mask is not None:
            self.processed_pil_image = temp_processed_pil
            self.cleaned_mask_auto = temp_cleaned_mask
            self.pil_mask_for_display = None
            self.manual_mask = None
            self.brick_contours = None
            self.processed_pil_with_contours = None
            self.processed_pil_contours_only = None

            self.reset_eraser_button.config(state=tk.DISABLED)
            self.export_dxf_button.config(state=tk.DISABLED)

            if self.erasing_active:
                self._toggle_eraser()
            if self.show_contours_active:
                self._toggle_show_contours(force_off=True)

            self.display_image(self.processed_canvas, self.processed_pil_image, "processed_photo", is_processed=True)
            self.show_contours_button.config(state=tk.NORMAL)
            self.show_binary_mask_button.config(state=tk.NORMAL)
        else:
            self._set_status("Errore applicazione filtri")
            self.display_image(self.processed_canvas, None, "processed_photo", is_processed=True)
            self.cleaned_mask_auto = None
            self.processed_pil_image = None
            self.pil_mask_for_display = None
            self.show_binary_mask_button.config(state=tk.DISABLED)
            self._close_binary_mask_window_if_open()

    def _toggle_eraser(self):
        if self.cleaned_mask_auto is None and self.manual_mask is None:
            return

        if self.show_contours_active:
            self._toggle_show_contours(force_off=True)

        self.erasing_active = not self.erasing_active

        if self.erasing_active:
            self._close_binary_mask_window_if_open()
            self.eraser_button.config(text="✏️ Disattiva")
            self.processed_canvas.config(cursor="dotbox")
            self._set_status("Modalità gomma attiva", "Click e trascina per cancellare")
            self._set_filter_controls_state(tk.DISABLED)
            self.show_contours_button.config(state=tk.DISABLED)
            self.export_dxf_button.config(state=tk.DISABLED)
            self.toggle_bg_button.config(state=tk.DISABLED)
            self.show_binary_mask_button.config(state=tk.DISABLED)
            self.reset_eraser_button.config(state=tk.NORMAL if self.manual_mask is not None else tk.DISABLED)

            self.processed_canvas.bind("<Button-1>", self._on_erase_press)
            self.processed_canvas.bind("<B1-Motion>", self._on_erase_motion)

            if self.manual_mask is None and self.cleaned_mask_auto is not None:
                self.manual_mask = self.cleaned_mask_auto.copy()
        else:
            self.eraser_button.config(text="✏️ Gomma")
            self.processed_canvas.config(cursor="arrow")
            self._set_status("Modalità gomma disattivata")

            if self.cleaned_mask_auto is not None or self.manual_mask is not None:
                self._set_filter_controls_state(tk.NORMAL)
                self.show_contours_button.config(state=tk.NORMAL)
                self.show_binary_mask_button.config(state=tk.NORMAL)

            self.processed_canvas.unbind("<Button-1>")
            self.processed_canvas.unbind("<B1-Motion>")

    def _reset_manual_edits(self):
        self._close_binary_mask_window_if_open()
        if self.cleaned_mask_auto is not None:
            self.manual_mask = None
            self.pil_mask_for_display = None
            self._redraw_processed_image_from_mask(self.cleaned_mask_auto)
            self.reset_eraser_button.config(state=tk.DISABLED)
            self.brick_contours = None
            self.processed_pil_with_contours = None
            self.processed_pil_contours_only = None
            self.export_dxf_button.config(state=tk.DISABLED)
            if self.show_contours_active:
                self._toggle_show_contours(force_off=True)
            self._set_status("Modifiche manuali annullate")
        else:
            messagebox.showwarning("Reset", "Nessuna maschera automatica a cui resettare.")

    def _canvas_to_mask_coords(self, event_x, event_y):
        params = self.processed_display_params
        img_x = event_x - params['x_offset']
        img_y = event_y - params['y_offset']

        if not (0 <= img_x < params['resized_w'] and 0 <= img_y < params['resized_h']):
            return None, None

        original_col = int(img_x * (params['original_w'] / params['resized_w']))
        original_row = int(img_y * (params['original_h'] / params['resized_h']))

        original_row = max(0, min(params['original_h'] - 1, original_row))
        original_col = max(0, min(params['original_w'] - 1, original_col))

        return original_row, original_col

    def _apply_eraser(self, event):
        if self.manual_mask is None:
            if self.cleaned_mask_auto is not None:
                self.manual_mask = self.cleaned_mask_auto.copy()
                self.reset_eraser_button.config(state=tk.NORMAL)
            else:
                return False

        row, col = self._canvas_to_mask_coords(event.x, event.y)
        if row is not None and col is not None:
            self._close_binary_mask_window_if_open()
            self.pil_mask_for_display = None

            brush_half = self.eraser_brush_size // 2
            r_start = max(0, row - brush_half)
            r_end = min(self.manual_mask.shape[0], row + brush_half + 1)
            c_start = max(0, col - brush_half)
            c_end = min(self.manual_mask.shape[1], col + brush_half + 1)

            self.manual_mask[r_start:r_end, c_start:c_end] = MORTAR_MASK_VALUE
            self.reset_eraser_button.config(state=tk.NORMAL)
            self.brick_contours = None
            self.processed_pil_with_contours = None
            self.processed_pil_contours_only = None
            self.export_dxf_button.config(state=tk.DISABLED)

            if self.show_contours_active:
                self._toggle_show_contours(force_off=True)

            return True
        return False

    def _on_erase_press(self, event):
        if self._apply_eraser(event):
            self._redraw_processed_image_from_mask(self.manual_mask)

    def _on_erase_motion(self, event):
        if self._apply_eraser(event):
            self._redraw_processed_image_from_mask(self.manual_mask)

    def _redraw_processed_image_from_mask(self, mask_to_use):
        if mask_to_use is None or self.img_lab is None:
            return

        temp_processed_pil = recolor_from_mask(mask_to_use, self.img_lab,
                                              TARGET_BRICK_COLOR_RGB, TARGET_MORTAR_COLOR_RGB)
        if temp_processed_pil:
            self.processed_pil_image = temp_processed_pil
            self.display_image(self.processed_canvas, self.processed_pil_image, "processed_photo", is_processed=True)

    def _get_final_mask(self):
        if self.manual_mask is not None:
            return self.manual_mask
        elif self.cleaned_mask_auto is not None:
            return self.cleaned_mask_auto
        else:
            return None

    def _find_and_prepare_contour_images(self):
        final_mask = self._get_final_mask()
        base_image_to_draw_on = self.processed_pil_image

        if final_mask is None or base_image_to_draw_on is None:
            messagebox.showerror("Errore", "Maschera o immagine non disponibili per i contorni.")
            return False

        self._set_status("Ricerca contorni...")

        try:
            if np.sum(final_mask == BRICK_MASK_VALUE) < np.sum(final_mask == MORTAR_MASK_VALUE):
                mask_for_cv_contours = BRICK_MASK_VALUE - final_mask
            else:
                mask_for_cv_contours = final_mask.copy()

            contours, _ = cv2.findContours(mask_for_cv_contours, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            self.brick_contours = contours

            img_with_contours_bgr = cv2.cvtColor(np.array(base_image_to_draw_on), cv2.COLOR_RGB2BGR)
            if self.brick_contours:
                cv2.drawContours(img_with_contours_bgr, self.brick_contours, -1, CONTOUR_COLOR_BGR, 1)
            self.processed_pil_with_contours = Image.fromarray(cv2.cvtColor(img_with_contours_bgr, cv2.COLOR_BGR2RGB))

            h, w = final_mask.shape
            contours_only_bgr = np.full((h, w, 3), BACKGROUND_COLOR_GRAY, dtype=np.uint8)
            if self.brick_contours:
                cv2.drawContours(contours_only_bgr, self.brick_contours, -1, CONTOUR_COLOR_BGR, 1)
            self.processed_pil_contours_only = Image.fromarray(cv2.cvtColor(contours_only_bgr, cv2.COLOR_BGR2RGB))

            self._set_status(f"Trovati {len(self.brick_contours)} contorni")
            self.export_dxf_button.config(state=tk.NORMAL if self.brick_contours else tk.DISABLED)

            return True
        except Exception as e:
            messagebox.showerror("Errore", f"Errore ricerca contorni:\n{e}")
            self.brick_contours = None
            self.processed_pil_with_contours = None
            self.processed_pil_contours_only = None
            self.export_dxf_button.config(state=tk.DISABLED)
            self._set_status("Errore contorni")
            return False

    def _toggle_show_contours(self, force_off=False):
        if self.erasing_active:
            messagebox.showwarning("Attenzione", "Disattiva prima la modalità gomma.")
            return

        current_mask = self._get_final_mask()
        if current_mask is None or self.processed_pil_image is None:
            messagebox.showwarning("Attenzione", "Elabora prima un'immagine.")
            return

        if force_off:
            if not self.show_contours_active:
                return
            self.show_contours_active = False
        else:
            self.show_contours_active = not self.show_contours_active

        if self.show_contours_active:
            if self.brick_contours is None:
                if not self._find_and_prepare_contour_images():
                    self.show_contours_active = False
                    return

            self.show_background_with_contours = True
            self.toggle_bg_button.config(state=tk.NORMAL)

            if self.processed_pil_with_contours:
                self.display_image(self.processed_canvas, self.processed_pil_with_contours,
                                 "processed_photo", is_processed=True)
                self.show_contours_button.config(text="📐 Nascondi")
                self._set_status("Bordi evidenziati")
        else:
            self.display_image(self.processed_canvas, self.processed_pil_image,
                             "processed_photo", is_processed=True)
            self.show_contours_button.config(text="📐 Bordi")
            self.toggle_bg_button.config(state=tk.DISABLED)
            self.show_background_with_contours = True
            self._set_status("Bordi nascosti")

    def _toggle_background_visibility(self):
        if not self.show_contours_active:
            return

        self.show_background_with_contours = not self.show_background_with_contours

        if self.show_background_with_contours:
            if self.processed_pil_with_contours:
                self.display_image(self.processed_canvas, self.processed_pil_with_contours,
                                 "processed_photo", is_processed=True)
                self.toggle_bg_button.config(text="🎨 Nascondi")
        else:
            if self.processed_pil_contours_only:
                self.display_image(self.processed_canvas, self.processed_pil_contours_only,
                                 "processed_photo", is_processed=True)
                self.toggle_bg_button.config(text="🎨 Mostra")

    def _get_scale_factor(self):
        try:
            factor = float(self.scale_factor_var.get().replace(',', '.'))
            if factor <= 0:
                messagebox.showwarning("Scala non valida", "Il fattore di scala deve essere positivo.")
                return None
            return factor
        except ValueError:
            messagebox.showerror("Errore", "Inserire un valore numerico per la scala.")
            return None

    def _export_dxf(self):
        if ezdxf is None:
            messagebox.showerror("Errore", "La libreria 'ezdxf' non è installata.")
            return

        if self.brick_contours is None:
            if not self._find_and_prepare_contour_images():
                messagebox.showerror("Errore", "Bordi non trovati.")
                return

        if not self.brick_contours:
            messagebox.showerror("Errore", "Nessun bordo valido trovato.")
            return

        scale_factor = self._get_scale_factor()
        if scale_factor is None:
            return

        dxf_path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")],
            title="Salva Bordi DXF"
        )
        if not dxf_path:
            return

        self._set_status("Esportazione DXF...", f"{len(self.brick_contours)} contorni")

        try:
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            layer_name = "Bordi_Mattoni"

            if layer_name not in doc.layers:
                doc.layers.add(name=layer_name, color=3)

            count = 0
            final_mask_for_height = self._get_final_mask()
            if final_mask_for_height is None:
                messagebox.showerror("Errore", "Impossibile determinare l'altezza dell'immagine.")
                return

            img_h = final_mask_for_height.shape[0]

            for contour in self.brick_contours:
                if len(contour) >= 2:
                    points = [(p[0][0] * scale_factor, (img_h - p[0][1]) * scale_factor) for p in contour]
                    msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
                    count += 1

            doc.saveas(dxf_path)
            self._set_status(f"DXF esportato: {os.path.basename(dxf_path)}", f"{count} bordi salvati")
            messagebox.showinfo("Export DXF", f"Esportazione completata.\n{count} bordi salvati.")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore esportazione DXF:\n{e}")
            self._set_status("Errore export DXF")

    def _close_binary_mask_window_if_open(self):
        if self.binary_mask_window is not None:
            try:
                self.binary_mask_window.destroy()
            except tk.TclError:
                pass
            self.binary_mask_window = None
            self.binary_mask_window_photo = None
            self.pil_mask_for_display = None

    def _on_close_binary_mask_window(self):
        self._close_binary_mask_window_if_open()

    def _export_binary_mask_png(self):
        mask_from_getter = self._get_final_mask()
        if mask_from_getter is None:
            messagebox.showerror("Errore", "Nessuna maschera disponibile.")
            return

        corrected_mask_for_export = BRICK_MASK_VALUE - mask_from_getter

        png_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Salva Maschera Binaria"
        )
        if not png_path:
            return

        try:
            pil_image_to_save = Image.fromarray(corrected_mask_for_export.astype(np.uint8), mode='L')
            pil_image_to_save.save(png_path)
            messagebox.showinfo("Export", f"Maschera salvata:\n{png_path}")
            self._set_status(f"Maschera esportata: {os.path.basename(png_path)}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare la maschera:\n{e}")

    def _show_binary_mask_window(self):
        final_mask_from_getter = self._get_final_mask()
        if final_mask_from_getter is None:
            messagebox.showwarning("Maschera", "Nessuna maschera disponibile.")
            return

        mask_data_for_display = BRICK_MASK_VALUE - final_mask_from_getter

        self._close_binary_mask_window_if_open()
        self.pil_mask_for_display = Image.fromarray(mask_data_for_display.astype(np.uint8), mode='L')

        self.binary_mask_window = tk.Toplevel(self.root)
        self.binary_mask_window.title("Maschera Binaria (Mattoni: Bianco)")

        orig_w, orig_h = self.pil_mask_for_display.size
        MAX_WIN_W, MAX_WIN_H = 800, 600
        win_aspect_ratio = orig_w / orig_h if orig_h > 0 else 1
        geom_w = MAX_WIN_W
        geom_h = int(MAX_WIN_W / win_aspect_ratio) if win_aspect_ratio > 0 else MAX_WIN_W

        if geom_h > MAX_WIN_H:
            geom_h = MAX_WIN_H
            geom_w = int(MAX_WIN_H * win_aspect_ratio) if win_aspect_ratio > 0 else MAX_WIN_H

        geom_w = min(geom_w, orig_w + 40)
        geom_h = min(geom_h, orig_h + 40)
        button_bar_height = 50

        self.binary_mask_window.geometry(f"{geom_w}x{geom_h + button_bar_height}")
        self.binary_mask_window.minsize(200, 150 + button_bar_height)
        self.binary_mask_window.protocol("WM_DELETE_WINDOW", self._on_close_binary_mask_window)

        canvas_frame = ttk.Frame(self.binary_mask_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        mask_canvas = tk.Canvas(canvas_frame, bg="gray50")
        mask_canvas.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(self.binary_mask_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        export_button = ttk.Button(button_frame, text="Esporta PNG", command=self._export_binary_mask_png)
        export_button.pack(pady=5)

        def _redraw_mask_on_resize(event=None):
            if self.binary_mask_window is None or not self.binary_mask_window.winfo_exists():
                return
            if self.pil_mask_for_display is None:
                return

            canvas_w = mask_canvas.winfo_width()
            canvas_h = mask_canvas.winfo_height()

            if canvas_w <= 1 or canvas_h <= 1:
                return

            img_resized = self._resize_image_for_canvas(self.pil_mask_for_display, canvas_w, canvas_h,
                                                       Image.Resampling.NEAREST)
            mask_canvas.delete("all")

            if img_resized:
                self.binary_mask_window_photo = ImageTk.PhotoImage(img_resized)
                x_pos = max(0, (canvas_w - img_resized.width) // 2)
                y_pos = max(0, (canvas_h - img_resized.height) // 2)
                mask_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.binary_mask_window_photo)

        mask_canvas.bind("<Configure>", _redraw_mask_on_resize)
        self.binary_mask_window.update_idletasks()
        _redraw_mask_on_resize()


# --- Main ---
def main():
    if USE_TTKB:
        root = ttk.Window(themename="darkly")
    else:
        root = tk.Tk()

    app = PyWALLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
