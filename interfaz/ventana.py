# ============================================================
# Módulo: ventana.py
# ============================================================
# Ventana principal — tema oscuro con customtkinter.
#
# PÁGINA 1 — Normalización por Canal RGB
# ────────────────────────────────────────────────────────────
# Columna izquierda:
#   ┌─────────────────────┐
#   │  Imagen Original    │  ← estática (referencia)
#   │─────────────────────│
#   │  Resultado Normaliz │  ← se actualiza en tiempo real
#   │  [Cargar Imagen]    │
#   └─────────────────────┘
#
# Columnas R / G / B:
#   ┌─────────────────────────┐
#   │  Canal X  (título)      │
#   │  [img orig] [hist orig] │  ← estático
#   │─────────────────────────│
#   │  [img norm] [hist norm] │  ← actualiza en tiempo real
#   │  ○───────────────────○  │  ← RangeSlider min/max
#   │  Min: X   Max: Y        │
#   │  [Limpiar]              │
#   └─────────────────────────┘
#
# Todo el contenido cabe en pantalla sin scroll.
#
# PÁGINA 2 — Compresión y Binarización
# ────────────────────────────────────────────────────────────
#   Tres imágenes + controles (bloque, umbral) + guardar.

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
from pathlib import Path

# Soporta ejecutar este archivo directamente (python interfaz/ventana.py)
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Procesamiento de imagen
from procesamiento.canales      import separar_canales, unir_canales
from procesamiento.histograma   import calcular_histograma
from procesamiento.redistribuir import normalizar_canal
from procesamiento.compresion   import convertir_a_grises, comprimir_por_bloques
from procesamiento.threshold    import aplicar_threshold

# Componentes de interfaz
from interfaz.componentes import convertir_imagen_para_tkinter, RangeSlider


# ── Paleta de colores ──────────────────────────────────────────────────────
C_FONDO       = "#0f1117"
C_TARJETA     = "#1a1d27"
C_BORDE       = "#2d3152"
C_TITULO      = "#e2e8f0"
C_SUBTITULO   = "#8892a4"
C_PLACEHOLDER = "#141720"
C_DARK_AX     = "#141720"
C_TICK        = "#5a6478"
C_SPINE       = "#2d3152"

C_AZUL        = "#3b82f6"
C_AZUL_HOVER  = "#2563eb"
C_VERDE       = "#16a34a"
C_VERDE_HOVER = "#15803d"
C_GRIS_BTN    = "#374151"
C_GRIS_HOVER  = "#4b5563"

COLOR_CANAL  = {"R": "#ef4444", "G": "#22c55e", "B": "#3b82f6"}
NOMBRE_CANAL = {"R": "Canal R — Rojo", "G": "Canal G — Verde", "B": "Canal B — Azul"}

# ── Tamaños  ───────────────────────────────────────────────────────────────
# Columna izquierda (original + recombinada)
ANCHO_ORIG = 320
ALTO_ORIG  = 180

# Cada canal: imagen + histograma lado a lado, compactos
ANCHO_CH_IMG = 208   # imagen de canal
ALTO_CH_IMG  = 132   # altura imagen
# Histograma más grande y debajo de la imagen
HIST_W_IN   = 3.35
HIST_H_IN   = 1.95

# Ancho del RangeSlider (aprox. 2 × ANCHO_CH_IMG)
SLIDER_ANCHO = ANCHO_CH_IMG * 2 + 6   # ≈ 302 px

# Página 2
ANCHO_P2 = 400
ALTO_P2  = 295


# ============================================================
class VentanaPrincipal:
    """Controlador principal de la interfaz."""

    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.configure(bg=C_FONDO)

        # ── Estado ──────────────────────────────────────────────────────
        self.imagen_original    = None
        self.canales_orig       = {"R": None, "G": None, "B": None}
        self.canales_norm       = {"R": None, "G": None, "B": None}
        self.imagen_recombinada = None
        self.imagen_comprimida  = None
        self.imagen_binaria     = None
        self.y_max_hist         = 1      # escala Y fija para histogramas
        self.umbral_actual      = 128
        self.bloque_actual      = 2
        self.refs               = {}     # previene GC de PhotoImage

        # ── Widgets de canal (se llenan en _construir_pagina1) ───────────
        self._lbl_orig_canal   = {}   # imagen original por canal (estática)
        self._ax_hist_orig     = {}   # histograma original (estático)
        self._cv_hist_orig     = {}
        self._lbl_norm_canal   = {}   # imagen normalizada por canal (dinámica)
        self._ax_hist_norm     = {}   # histograma normalizado (dinámico, Y fijo)
        self._cv_hist_norm     = {}
        self._sliders          = {}   # RangeSlider por canal
        self._lbl_minmax       = {}   # etiqueta Min/Max

        self._construir_ui()

    # ═══════════════════════════════════════════════════════════════════════
    # CONSTRUCCIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_ui(self):
        self._contenedor = ctk.CTkFrame(self.raiz, fg_color=C_FONDO)
        self._contenedor.pack(fill="both", expand=True)
        self._pagina1 = ctk.CTkScrollableFrame(self._contenedor, fg_color=C_FONDO)
        self._pagina2 = ctk.CTkScrollableFrame(self._contenedor, fg_color=C_FONDO)
        self._construir_pagina1()
        self._construir_pagina2()
        self._ir_pagina1()

    # ── Barra de navegación ──────────────────────────────────────────────

    def _construir_barra_nav(self):
        barra = ctk.CTkFrame(self.raiz, fg_color=C_TARJETA, height=46, corner_radius=0)
        barra.pack(fill="x")
        barra.pack_propagate(False)

        ctk.CTkLabel(barra,
                     text="Ecualizador de Imágenes — Preprocesamiento ML",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C_TITULO).pack(side="left", padx=16, pady=10)

        frame_bc = ctk.CTkFrame(barra, fg_color=C_TARJETA)
        frame_bc.pack(side="right", padx=12)

        self._btn_nav1 = ctk.CTkButton(frame_bc, text="1 · Normalización",
                                        width=150, height=28,
                                        font=ctk.CTkFont(size=11),
                                        fg_color=C_AZUL,
                                        command=self._ir_pagina1)
        self._btn_nav1.pack(side="left", padx=(0, 4))

        self._btn_nav2 = ctk.CTkButton(frame_bc, text="2 · Comprimir y Binarizar",
                                        width=178, height=28,
                                        font=ctk.CTkFont(size=11),
                                        fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
                                        command=self._ir_pagina2)
        self._btn_nav2.pack(side="left")

    # ── Navegación ───────────────────────────────────────────────────────

    def _ir_pagina1(self):
        self._pagina2.pack_forget()
        self._pagina1.pack(fill="both", expand=True)

    def _ir_pagina2(self):
        if self.imagen_original is None:
            messagebox.showwarning("Sin imagen",
                                   "Primero carga una imagen en la página de Normalización.")
            return
        self._pagina1.pack_forget()
        self._pagina2.pack(fill="both", expand=True)
        self._actualizar_pagina2()

    # ═══════════════════════════════════════════════════════════════════════
    # PÁGINA 1 — Normalización
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_pagina1(self):
        p = self._pagina1

        ctk.CTkLabel(p,
                     text="Paso 1 de 2 — Normalización por Canal RGB",
                     font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=C_TITULO).pack(pady=(10, 1))

        ctk.CTkLabel(p,
                     text=(
                         "Original arriba · Normalizado abajo.  "
                         "Arrastra el slider de cada canal para ajustar el rango."
                     ),
                     font=ctk.CTkFont(size=12),
                     text_color=C_SUBTITULO).pack(pady=(0, 8))

        # ── Fila de 4 columnas ────────────────────────────────────────────
        fila = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila.pack(fill="x", padx=8)

        self._construir_col_original(fila)
        for letra in ("R", "G", "B"):
            self._construir_col_canal(fila, letra)

        # ── Botones de acción ────────────────────────────────────────────
        fila_acciones = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_acciones.pack(pady=10)

        ctk.CTkButton(fila_acciones,
                      text="↺  Reset Todo",
                      fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
                      font=ctk.CTkFont(size=12), height=36, width=180,
                      command=self._reset_todo).pack(side="left", padx=6)

        ctk.CTkButton(fila_acciones,
                      text="→  Siguiente: Comprimir y Binarizar",
                      fg_color=C_AZUL, hover_color=C_AZUL_HOVER,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      height=36, width=320,
                      command=self._ir_pagina2).pack(side="left", padx=6)

    # ── Columna izquierda: Original + Recombinada ────────────────────────

    def _construir_col_original(self, contenedor):
        col = ctk.CTkFrame(contenedor, fg_color=C_TARJETA,
                           corner_radius=10, border_color=C_BORDE, border_width=1)
        col.pack(side="left", padx=5, pady=6, anchor="n")

        # Imagen original (referencia fija)
        ctk.CTkLabel(col, text="Imagen Original",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C_TITULO).pack(pady=(8, 2))
        self._lbl_original = self._cuadro_imagen(col, ANCHO_ORIG, ALTO_ORIG,
                                                  "Carga\nuna imagen")

        ctk.CTkFrame(col, height=1, fg_color=C_BORDE).pack(fill="x", padx=10, pady=5)

        # Imagen recombinada (resultado en tiempo real)
        ctk.CTkLabel(col, text="↓ Resultado Normalizado",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#22c55e").pack(pady=(0, 2))
        self._lbl_recombinada = self._cuadro_imagen(col, ANCHO_ORIG, ALTO_ORIG,
                                                     "Sin imagen\nnormalizada")

        ctk.CTkButton(col, text="📂  Cargar Imagen",
                      fg_color=C_AZUL, hover_color=C_AZUL_HOVER,
                      height=38, width=170,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      command=self._cargar_imagen).pack(pady=(10, 12), padx=10)

    # ── Columna de canal (R / G / B) ─────────────────────────────────────

    def _construir_col_canal(self, contenedor, letra):
        """
        Estructura de la columna:
            Título del canal
            ──────────────────────────────
            ORIGINAL (estático)
              [imagen gris orig] [hist orig]
            ──────────────────────────────
            NORMALIZADO (tiempo real)
              [imagen gris norm] [hist norm]
              ○────────────────────────────○   ← RangeSlider
              Min: X   Max: Y
            [Limpiar]
        """
        color = COLOR_CANAL[letra]
        col   = ctk.CTkFrame(contenedor, fg_color=C_TARJETA,
                              corner_radius=10, border_color=C_BORDE, border_width=1)
        col.pack(side="left", padx=5, pady=6, anchor="n")

        ctk.CTkLabel(col, text=NOMBRE_CANAL[letra],
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=color).pack(pady=(8, 2))

        # ── Banda ORIGINAL ────────────────────────────────────────────
        ctk.CTkLabel(col, text="ORIGINAL",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=C_SUBTITULO).pack()

        fila_orig = tk.Frame(col, bg=C_TARJETA)
        fila_orig.pack(padx=10, pady=(3, 6))

        lbl_orig = self._cuadro_imagen_inline(fila_orig, ANCHO_CH_IMG, ALTO_CH_IMG, "—",
                                              side="top")
        self._lbl_orig_canal[letra] = lbl_orig

        fig_o, ax_o = self._nueva_figura_hist()
        cv_o = FigureCanvasTkAgg(fig_o, master=fila_orig)
        cv_o.get_tk_widget().pack(side="top", pady=(6, 0))
        self._ax_hist_orig[letra] = ax_o
        self._cv_hist_orig[letra] = cv_o

        # ── RangeSlider (debajo de ORIGINAL) ─────────────────────────
        slider = RangeSlider(
            col,
            desde=0, hasta=255,
            valor_min_inicial=0, valor_max_inicial=255,
            ancho=SLIDER_ANCHO,
            color_activo=color,
            color_fondo=C_TARJETA,
            callback=lambda vmin, vmax, l=letra: self._al_cambiar_slider(l, vmin, vmax)
        )
        slider.pack(pady=(1, 8))
        self._sliders[letra] = slider

        # Separador
        ctk.CTkFrame(col, height=1, fg_color=C_BORDE).pack(fill="x", padx=10, pady=4)

        # ── Banda NORMALIZADO ─────────────────────────────────────────
        ctk.CTkLabel(col, text="NORMALIZADO",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color=color).pack()

        fila_norm = tk.Frame(col, bg=C_TARJETA)
        fila_norm.pack(padx=10, pady=(3, 6))

        lbl_norm = self._cuadro_imagen_inline(fila_norm, ANCHO_CH_IMG, ALTO_CH_IMG, "—",
                                               side="top")
        self._lbl_norm_canal[letra] = lbl_norm

        fig_n, ax_n = self._nueva_figura_hist()
        cv_n = FigureCanvasTkAgg(fig_n, master=fila_norm)
        cv_n.get_tk_widget().pack(side="top", pady=(6, 0))
        self._ax_hist_norm[letra] = ax_n
        self._cv_hist_norm[letra] = cv_n

        # Botón limpiar
        ctk.CTkButton(col, text=f"↺  Limpiar {letra}",
                      fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
                      height=32, width=140, font=ctk.CTkFont(size=11),
                      command=lambda l=letra: self._reset_canal(l)).pack(pady=(2, 12))

    # ═══════════════════════════════════════════════════════════════════════
    # PÁGINA 2 — Compresión y Binarización
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_pagina2(self):
        p = self._pagina2

        ctk.CTkLabel(p,
                     text="Paso 2 de 2 — Compresión por Bloques y Binarización",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=C_TITULO).pack(pady=(14, 2))
        ctk.CTkLabel(p,
                     text="Se aplica sobre la imagen ya normalizada del paso anterior.",
                     font=ctk.CTkFont(size=10),
                     text_color=C_SUBTITULO).pack(pady=(0, 12))

        # ── Tres imágenes ─────────────────────────────────────────────────
        fila_imgs = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_imgs.pack(pady=4, padx=10)

        for titulo, attr, attr_dim in [
            ("Original Normalizada",           "_lbl_p2_orig", "_lbl_dim_orig"),
            ("Imagen Comprimida (grises)",      "_lbl_p2_comp", "_lbl_dim_comp"),
            ("Imagen Binaria (Threshold)",      "_lbl_p2_bin",  "_lbl_info_bin"),
        ]:
            tarj = ctk.CTkFrame(fila_imgs, fg_color=C_TARJETA,
                                corner_radius=10, border_color=C_BORDE, border_width=1)
            tarj.pack(side="left", padx=6, pady=4)
            ctk.CTkLabel(tarj, text=titulo,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=C_TITULO).pack(pady=(8, 2))
            lbl = self._cuadro_imagen(tarj, ANCHO_P2, ALTO_P2, "—")
            setattr(self, attr, lbl)
            lbl_info = ctk.CTkLabel(tarj, text="",
                                    font=ctk.CTkFont(size=10),
                                    text_color=C_SUBTITULO)
            lbl_info.pack(pady=(2, 8))
            setattr(self, attr_dim, lbl_info)

        # ── Panel de controles ────────────────────────────────────────────
        panel = ctk.CTkFrame(p, fg_color=C_TARJETA,
                              corner_radius=10, border_color=C_BORDE, border_width=1)
        panel.pack(fill="x", padx=18, pady=12)

        ctk.CTkLabel(panel, text="Controles de Procesamiento",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TITULO).pack(anchor="w", padx=14, pady=(10, 4))
        ctk.CTkFrame(panel, height=1, fg_color=C_BORDE).pack(fill="x", padx=14)

        # Tamaño de bloque
        sb = ctk.CTkFrame(panel, fg_color=C_TARJETA)
        sb.pack(anchor="w", padx=14, pady=(8, 4))
        ctk.CTkLabel(sb, text="Tamaño de bloque (promedio NxN):",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_TITULO).pack(anchor="w", pady=(0, 4))

        fila_rb = ctk.CTkFrame(sb, fg_color=C_TARJETA)
        fila_rb.pack(anchor="w")
        self._var_bloque = ctk.IntVar(value=2)
        for tam in (2, 4, 8, 16):
            ctk.CTkRadioButton(fila_rb, text=f"  {tam}×{tam}  ",
                               variable=self._var_bloque, value=tam,
                               command=self._al_cambiar_bloque,
                               text_color=C_TITULO,
                               radiobutton_width=15, radiobutton_height=15,
                               font=ctk.CTkFont(size=11)).pack(side="left", padx=10)

        ctk.CTkFrame(panel, height=1, fg_color=C_BORDE).pack(fill="x", padx=14, pady=6)

        # Umbral
        su = ctk.CTkFrame(panel, fg_color=C_TARJETA)
        su.pack(fill="x", padx=14, pady=(0, 10))

        fila_tit = ctk.CTkFrame(su, fg_color=C_TARJETA)
        fila_tit.pack(fill="x")
        ctk.CTkLabel(fila_tit, text="Umbral de binarización (threshold):",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_TITULO).pack(side="left", pady=(0, 4))
        self._lbl_media_ref = ctk.CTkLabel(fila_tit,
                                            text="   |   Media: —",
                                            font=ctk.CTkFont(size=10),
                                            text_color=C_SUBTITULO)
        self._lbl_media_ref.pack(side="left", pady=(0, 4))

        fila_sl = ctk.CTkFrame(su, fg_color=C_TARJETA)
        fila_sl.pack(fill="x")
        self._slider_umbral = ctk.CTkSlider(fila_sl, from_=0, to=255,
                                             number_of_steps=255, width=380,
                                             command=self._al_cambiar_umbral)
        self._slider_umbral.set(128)
        self._slider_umbral.pack(side="left", padx=(0, 10))
        self._lbl_umbral_val = ctk.CTkLabel(fila_sl, text="Umbral: 128",
                                             font=ctk.CTkFont(family="Consolas",
                                                              size=11, weight="bold"),
                                             text_color=C_TITULO, width=110)
        self._lbl_umbral_val.pack(side="left")

        ctk.CTkLabel(su, text="pixel ≥ umbral → 255 (blanco)   |   pixel < umbral → 0 (negro)",
                     font=ctk.CTkFont(size=9), text_color=C_SUBTITULO).pack(anchor="w", pady=(4, 0))

        # ── Navegación ────────────────────────────────────────────────────
        fila_nav = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_nav.pack(fill="x", padx=18, pady=(6, 18))
        ctk.CTkButton(fila_nav, text="← Volver",
                      fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER, height=34,
                      command=self._ir_pagina1).pack(side="left")
        ctk.CTkButton(fila_nav, text="💾  Guardar Imagen Binaria",
                      fg_color=C_VERDE, hover_color=C_VERDE_HOVER, height=34,
                      command=self._guardar_binaria).pack(side="right")

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Carga de imagen
    # ═══════════════════════════════════════════════════════════════════════

    def _cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("Todos", "*.*")]
        )
        if not ruta:
            return
        img = cv2.imread(ruta)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return

        self.imagen_original = img
        r, g, b = separar_canales(img)
        self.canales_orig = {"R": r, "G": g, "B": b}

        # Histogramas originales + escala Y fija para TODOS los histogramas
        hists_orig = {l: calcular_histograma(self.canales_orig[l]) for l in "RGB"}
        self.y_max_hist = max(h.max() for h in hists_orig.values())
        if self.y_max_hist < 1:
            self.y_max_hist = 1

        # Mostrar imagen original y los histogramas/imágenes originales (estáticos)
        self._mostrar(self._lbl_original, img, "orig", ANCHO_ORIG, ALTO_ORIG)

        for letra in "RGB":
            # Imagen original del canal
            self._mostrar(self._lbl_orig_canal[letra],
                          self.canales_orig[letra],
                          f"orig_ch_{letra}", ANCHO_CH_IMG, ALTO_CH_IMG)
            # Histograma original (estático, se dibuja una sola vez)
            self._dibujar_hist(self._ax_hist_orig[letra],
                               self._cv_hist_orig[letra],
                               hists_orig[letra],
                               COLOR_CANAL[letra])

            # Resetear slider silenciosamente (sin disparar callback)
            self._sliders[letra].set_valores(0, 255)

            # Canal normalizado = copia del original (min=0, max=255 es identidad)
            self.canales_norm[letra] = self.canales_orig[letra].copy()
            self._mostrar(self._lbl_norm_canal[letra],
                          self.canales_norm[letra],
                          f"norm_ch_{letra}", ANCHO_CH_IMG, ALTO_CH_IMG)
            self._dibujar_hist(self._ax_hist_norm[letra],
                               self._cv_hist_norm[letra],
                               hists_orig[letra],
                               COLOR_CANAL[letra])

        self._actualizar_recombinada()

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Normalización en tiempo real
    # ═══════════════════════════════════════════════════════════════════════

    def _al_cambiar_slider(self, letra, vmin, vmax):
        """Callback del RangeSlider: normaliza el canal y actualiza vistas."""
        if self.imagen_original is None:
            return
        self._normalizar_canal(letra, vmin, vmax)
        self._actualizar_recombinada()

    def _normalizar_canal(self, letra, vmin, vmax):
        """Aplica normalización y refresca imagen + histograma normalizado."""
        canal_norm = normalizar_canal(self.canales_orig[letra], vmin, vmax)
        self.canales_norm[letra] = canal_norm

        self._mostrar(self._lbl_norm_canal[letra],
                      canal_norm, f"norm_ch_{letra}", ANCHO_CH_IMG, ALTO_CH_IMG)

        hist = calcular_histograma(canal_norm)
        self._dibujar_hist(self._ax_hist_norm[letra],
                           self._cv_hist_norm[letra],
                           hist, COLOR_CANAL[letra])

    def _actualizar_recombinada(self):
        if any(v is None for v in self.canales_norm.values()):
            return
        self.imagen_recombinada = unir_canales(
            self.canales_norm["R"], self.canales_norm["G"], self.canales_norm["B"]
        )
        self._mostrar(self._lbl_recombinada, self.imagen_recombinada,
                      "recomb", ANCHO_ORIG, ALTO_ORIG)

    # ── Resets ───────────────────────────────────────────────────────────

    def _reset_canal(self, letra):
        self._sliders[letra].set_valores(0, 255)
        if self.imagen_original is not None:
            self._normalizar_canal(letra, 0, 255)
            self._actualizar_recombinada()

    def _reset_todo(self):
        for letra in "RGB":
            self._reset_canal(letra)

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Página 2
    # ═══════════════════════════════════════════════════════════════════════

    def _actualizar_pagina2(self):
        if self.imagen_recombinada is None:
            return
        h, w = self.imagen_recombinada.shape[:2]
        self._mostrar(self._lbl_p2_orig, self.imagen_recombinada,
                      "p2_orig", ANCHO_P2, ALTO_P2)
        self._lbl_dim_orig.configure(text=f"{w} × {h} px")
        self._aplicar_compresion_y_threshold()

    def _al_cambiar_bloque(self):
        self.bloque_actual = self._var_bloque.get()
        if self.imagen_recombinada is not None:
            self._aplicar_compresion_y_threshold()

    def _al_cambiar_umbral(self, valor):
        self.umbral_actual = int(float(valor))
        self._lbl_umbral_val.configure(text=f"Umbral: {self.umbral_actual}")
        if self.imagen_comprimida is not None:
            self._aplicar_threshold_solo()

    def _aplicar_compresion_y_threshold(self):
        gris = convertir_a_grises(self.imagen_recombinada)
        comp = comprimir_por_bloques(gris, self.bloque_actual)
        self.imagen_comprimida = comp
        h, w = comp.shape[:2]
        self._mostrar(self._lbl_p2_comp, comp, "p2_comp", ANCHO_P2, ALTO_P2)
        self._lbl_dim_comp.configure(
            text=f"Bloque {self.bloque_actual}×{self.bloque_actual}  |  {w}×{h} px")
        self._aplicar_threshold_solo()

    def _aplicar_threshold_solo(self):
        if self.imagen_comprimida is None:
            return
        binaria, media, umbral_usado = aplicar_threshold(
            self.imagen_comprimida, self.umbral_actual)
        self.imagen_binaria = binaria
        self._mostrar(self._lbl_p2_bin, binaria, "p2_bin", ANCHO_P2, ALTO_P2)
        self._lbl_info_bin.configure(
            text=f"Umbral: {umbral_usado}  |  Media: {media:.1f}")
        self._lbl_media_ref.configure(
            text=f"   |   Media: {media:.1f}")

    def _guardar_binaria(self):
        if self.imagen_binaria is None:
            messagebox.showwarning("Sin imagen",
                                   "Primero genera la imagen binaria en esta página.")
            return
        os.makedirs("imagenes", exist_ok=True)
        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen binaria", initialdir="imagenes",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")])
        if ruta:
            cv2.imwrite(ruta, self.imagen_binaria)
            messagebox.showinfo("Guardado", f"Imagen guardada en:\n{ruta}")

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS DE INTERFAZ
    # ═══════════════════════════════════════════════════════════════════════

    def _cuadro_imagen(self, contenedor, ancho, alto, texto=""):
        """Frame fijo con Label para imágenes. Empaqueta con pack()."""
        marco = ctk.CTkFrame(contenedor, width=ancho, height=alto,
                              fg_color=C_PLACEHOLDER, corner_radius=4,
                              border_color=C_BORDE, border_width=1)
        marco.pack(pady=4, padx=8)
        marco.pack_propagate(False)
        lbl = tk.Label(marco, text=texto,
                       bg=C_PLACEHOLDER, fg=C_SUBTITULO,
                       font=("Arial", 8, "italic"),
                       wraplength=ancho - 12, anchor="center")
        lbl.pack(expand=True)
        return lbl

    def _cuadro_imagen_inline(self, contenedor, ancho, alto, texto="", side="left"):
        """Frame fijo empaquetado con pack(side=side) para layout horizontal."""
        marco = ctk.CTkFrame(contenedor, width=ancho, height=alto,
                              fg_color=C_PLACEHOLDER, corner_radius=4,
                              border_color=C_BORDE, border_width=1)
        marco.pack(side=side)
        marco.pack_propagate(False)
        lbl = tk.Label(marco, text=texto,
                       bg=C_PLACEHOLDER, fg=C_SUBTITULO,
                       font=("Arial", 7, "italic"),
                       wraplength=ancho - 8, anchor="center")
        lbl.pack(expand=True)
        return lbl

    def _mostrar(self, label, imagen_numpy, clave, ancho_max, alto_max):
        """Convierte numpy → PhotoImage y lo asigna al Label."""
        if imagen_numpy is None:
            return
        img_tk = convertir_imagen_para_tkinter(imagen_numpy, ancho_max, alto_max)
        if img_tk is None:
            return
        self.refs[clave] = img_tk
        label.config(image=img_tk, text="")

    def _nueva_figura_hist(self):
        """Crea una figura matplotlib oscura del tamaño de las imágenes de canal."""
        fig = Figure(figsize=(HIST_W_IN, HIST_H_IN), dpi=80, facecolor=C_DARK_AX)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C_DARK_AX)
        ax.tick_params(axis="x", labelsize=7, colors=C_TICK)
        ax.tick_params(axis="y", labelsize=8, colors=C_TICK, pad=6)
        for sp in ax.spines.values():
            sp.set_color(C_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 255)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.subplots_adjust(left=0.23, right=0.98, bottom=0.18, top=0.95)
        return fig, ax

    def _dibujar_hist(self, ax, canvas_mpl, frecuencias, color):
        """
        Dibuja el histograma con eje Y FIJO (self.y_max_hist).
        Usar la misma escala en los 6 histogramas (3 orig + 3 norm)
        permite comparar visualmente la redistribución de píxeles.
        """
        ax.clear()
        ax.set_facecolor(C_DARK_AX)
        ax.tick_params(axis="x", labelsize=7, colors=C_TICK)
        ax.tick_params(axis="y", labelsize=8, colors=C_TICK, pad=6)
        for sp in ax.spines.values():
            sp.set_color(C_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, self.y_max_hist * 1.05)   # eje Y fijo y constante
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.bar(range(256), frecuencias, color=color, width=1.0, alpha=0.85)
        ax.grid(True, alpha=0.10, color="white", linewidth=0.4)
        canvas_mpl.draw_idle()
