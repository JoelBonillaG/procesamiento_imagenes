# ============================================================
# Módulo: ventana.py
# ============================================================
# Ventana principal con customtkinter (tema oscuro moderno).
#
# PÁGINA 1 — Normalización por Canal RGB:
#   Cuatro columnas: [Imagen Original] [Canal R] [Canal G] [Canal B]
#   Cada canal muestra:
#     - Imagen en escala de grises (actualizada en tiempo real)
#     - Curva interactiva estilo Levels con histograma de fondo
#     - Valores Min / Max actuales
#     - Botón "Limpiar" para resetear solo ese canal
#   Debajo: imagen recombinada + canales normalizados con histogramas (eje Y fijo)
#
# PÁGINA 2 — Compresión y Binarización:
#   Tres imágenes lado a lado: Normalizada | Comprimida | Binaria
#   Panel de controles:
#     - Radio buttons para tamaño de bloque (2×2 / 4×4 / 8×8 / 16×16)
#     - Slider para el umbral de binarización (0–255)
#     - Media de la imagen mostrada como referencia
#
# Proceso secuencial:
#   Cargar → Normalizar por canal → Recombinar → Comprimir → Binarizar

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Módulos de procesamiento de imagen
from procesamiento.canales      import separar_canales, unir_canales
from procesamiento.histograma   import calcular_histograma
from procesamiento.redistribuir import normalizar_canal
from procesamiento.compresion   import convertir_a_grises, comprimir_por_bloques
from procesamiento.threshold    import aplicar_threshold

# Componentes de la interfaz
from interfaz.componentes import convertir_imagen_para_tkinter
from interfaz.curva_canal  import CurvaCanal


# ── Paleta de colores ──────────────────────────────────────────────────────
C_FONDO       = "#0f1117"   # fondo general de la ventana
C_TARJETA     = "#1a1d27"   # fondo de las tarjetas / secciones
C_BORDE       = "#2d3152"   # borde sutil entre elementos
C_TITULO      = "#e2e8f0"   # texto principal blanco-azulado
C_SUBTITULO   = "#8892a4"   # texto secundario gris
C_PLACEHOLDER = "#141720"   # fondo de cuadros de imagen vacíos
C_DARK_AX     = "#141720"   # fondo de ejes de matplotlib
C_TICK        = "#5a6478"   # color de números en ejes matplotlib
C_SPINE       = "#2d3152"   # color de bordes de ejes matplotlib

C_AZUL        = "#3b82f6"   # azul principal (botones de acción)
C_AZUL_HOVER  = "#2563eb"
C_VERDE       = "#16a34a"   # verde para guardar
C_VERDE_HOVER = "#15803d"
C_GRIS_BTN    = "#374151"   # gris para botones secundarios
C_GRIS_HOVER  = "#4b5563"

# Color por canal de color
COLOR_CANAL  = {"R": "#ef4444", "G": "#22c55e", "B": "#3b82f6"}
NOMBRE_CANAL = {"R": "Canal R — Rojo", "G": "Canal G — Verde", "B": "Canal B — Azul"}

# ── Tamaños de imágenes ────────────────────────────────────────────────────
ANCHO_ORIG  = 290   # imagen original y recombinada (página 1)
ALTO_ORIG   = 215

ANCHO_CANAL = 280   # imagen de canal sobre la curva (actualización en tiempo real)
ALTO_CANAL  = 200

ANCHO_NORM  = 380   # imagen canal normalizado (sección inferior página 1)
ALTO_NORM   = 250

ANCHO_P2    = 400   # imágenes en página 2
ALTO_P2     = 295


# ============================================================
# Clase principal
# ============================================================

class VentanaPrincipal:
    """
    Controlador principal de la interfaz gráfica.
    Gestiona el estado de la aplicación y coordina las dos páginas.
    """

    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.configure(bg=C_FONDO)

        # ── Estado de la aplicación ───────────────────────────────────────
        self.imagen_original    = None   # imagen BGR cargada con cv2.imread
        self.canales_orig       = {"R": None, "G": None, "B": None}
        self.canales_norm       = {"R": None, "G": None, "B": None}
        self.imagen_recombinada = None
        self.imagen_comprimida  = None
        self.imagen_binaria     = None

        # Máximo de conteo en los histogramas originales.
        # Se usa para que el eje Y de los histogramas normalizados sea FIJO,
        # lo que permite comparar visualmente la redistribución de píxeles.
        self.y_max_hist = 1

        self.umbral_actual = 128   # valor del slider de threshold
        self.bloque_actual = 2     # tamaño de bloque de compresión

        # Guardamos referencias a los PhotoImage para que Python no los elimine
        # (sin esto las imágenes desaparecen aunque el Label las esté mostrando)
        self.refs = {}

        # ── Widgets de canal (se llenan en _construir_pagina1) ────────────
        self._lbl_canal_top  = {}   # imagen de canal sobre la curva (en tiempo real)
        self._curvas         = {}   # CurvaCanal interactiva por canal
        self._lbl_minmax     = {}   # etiqueta "Min: X   Max: Y"
        self._lbl_canal_norm = {}   # imagen canal normalizado (sección inferior)
        self._ax_hist_norm   = {}   # eje del histograma normalizado
        self._cv_hist_norm   = {}   # canvas (widget) del histograma normalizado

        # ── Construir la interfaz completa ────────────────────────────────
        self._construir_ui()

    # ═══════════════════════════════════════════════════════════════════════
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_ui(self):
        # Barra de navegación fija (no entra en el scroll)
        self._construir_barra_nav()

        # Contenedor donde se intercambian las dos páginas
        self._contenedor = ctk.CTkFrame(self.raiz, fg_color=C_FONDO)
        self._contenedor.pack(fill="both", expand=True)

        # Páginas como CTkScrollableFrame (incluye scroll vertical automático)
        self._pagina1 = ctk.CTkScrollableFrame(self._contenedor, fg_color=C_FONDO)
        self._pagina2 = ctk.CTkScrollableFrame(self._contenedor, fg_color=C_FONDO)

        self._construir_pagina1()
        self._construir_pagina2()

        self._ir_pagina1()

    # ── Barra de navegación superior ─────────────────────────────────────

    def _construir_barra_nav(self):
        barra = ctk.CTkFrame(self.raiz, fg_color=C_TARJETA, height=48, corner_radius=0)
        barra.pack(fill="x")
        barra.pack_propagate(False)

        ctk.CTkLabel(
            barra,
            text="Ecualizador de Imágenes — Preprocesamiento ML",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TITULO
        ).pack(side="left", padx=18, pady=10)

        # Breadcrumb (migas de pan) a la derecha
        frame_breadcrumb = ctk.CTkFrame(barra, fg_color=C_TARJETA)
        frame_breadcrumb.pack(side="right", padx=14)

        self._btn_nav1 = ctk.CTkButton(
            frame_breadcrumb,
            text="1 · Normalización",
            width=155, height=30,
            font=ctk.CTkFont(size=11),
            fg_color=C_AZUL,
            command=self._ir_pagina1
        )
        self._btn_nav1.pack(side="left", padx=(0, 4))

        self._btn_nav2 = ctk.CTkButton(
            frame_breadcrumb,
            text="2 · Comprimir y Binarizar",
            width=180, height=30,
            font=ctk.CTkFont(size=11),
            fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
            command=self._ir_pagina2
        )
        self._btn_nav2.pack(side="left")

    # ── Navegación entre páginas ──────────────────────────────────────────

    def _ir_pagina1(self):
        self._pagina2.pack_forget()
        self._pagina1.pack(fill="both", expand=True)
        self._btn_nav1.configure(fg_color=C_AZUL)
        self._btn_nav2.configure(fg_color=C_GRIS_BTN)

    def _ir_pagina2(self):
        if self.imagen_original is None:
            messagebox.showwarning(
                "Sin imagen",
                "Primero carga una imagen en la página de Normalización."
            )
            return
        self._pagina1.pack_forget()
        self._pagina2.pack(fill="both", expand=True)
        self._btn_nav1.configure(fg_color=C_GRIS_BTN)
        self._btn_nav2.configure(fg_color=C_AZUL)
        self._actualizar_pagina2()

    # ═══════════════════════════════════════════════════════════════════════
    # PÁGINA 1 — Normalización por Canal
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_pagina1(self):
        p = self._pagina1

        # Títulos de la página
        ctk.CTkLabel(
            p,
            text="Paso 1 de 2 — Normalización por Canal RGB",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=C_TITULO
        ).pack(pady=(16, 2))

        ctk.CTkLabel(
            p,
            text=(
                "Arrastra los puntos de cada curva para definir el rango de normalización.  "
                "La imagen del canal se actualiza en tiempo real."
            ),
            font=ctk.CTkFont(size=11),
            text_color=C_SUBTITULO
        ).pack(pady=(0, 12))

        # ── Fila superior: Imagen Original + Canal R + Canal G + Canal B ──
        fila_top = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_top.pack(fill="x", padx=8)

        self._construir_col_original(fila_top)
        for letra in ("R", "G", "B"):
            self._construir_col_canal(fila_top, letra)

        # ── Botón Reset Todo ──────────────────────────────────────────────
        ctk.CTkButton(
            p,
            text="↺   Reset Todo  (restaurar los 3 canales a sin normalización)",
            fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
            font=ctk.CTkFont(size=11),
            height=34, width=420,
            command=self._reset_todo
        ).pack(pady=12)

        # Separador visual
        ctk.CTkFrame(p, height=2, fg_color=C_BORDE).pack(fill="x", padx=16, pady=4)

        # ── Sección: Imagen Recombinada ───────────────────────────────────
        ctk.CTkLabel(
            p,
            text="Imagen Recombinada Normalizada",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_TITULO
        ).pack(pady=(12, 4))

        marco_recomb = ctk.CTkFrame(p, fg_color=C_TARJETA,
                                    corner_radius=8,
                                    border_color=C_BORDE, border_width=1)
        marco_recomb.pack(pady=4)

        self._lbl_recombinada = self._cuadro_imagen(
            marco_recomb, ANCHO_ORIG * 2, ALTO_ORIG,
            "Sin imagen normalizada aún"
        )

        # ── Sección: Canales Normalizados + Histogramas ───────────────────
        ctk.CTkLabel(
            p,
            text="Canales Normalizados — R' · G' · B'  (histogramas con eje Y fijo)",
            font=ctk.CTkFont(size=12),
            text_color=C_SUBTITULO
        ).pack(pady=(16, 4))

        ctk.CTkLabel(
            p,
            text=(
                "El eje Y de los histogramas usa la escala del canal original, "
                "permitiendo ver visualmente cómo se redistribuyen los píxeles."
            ),
            font=ctk.CTkFont(size=10),
            text_color=C_SUBTITULO
        ).pack(pady=(0, 6))

        fila_norm = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_norm.pack(pady=4)

        for letra in ("R", "G", "B"):
            self._construir_col_normalizado(fila_norm, letra)

        # ── Botón Siguiente ───────────────────────────────────────────────
        ctk.CTkFrame(p, height=2, fg_color=C_BORDE).pack(fill="x", padx=16, pady=12)

        ctk.CTkButton(
            p,
            text="→   Siguiente: Comprimir y Binarizar",
            fg_color=C_AZUL, hover_color=C_AZUL_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
            command=self._ir_pagina2
        ).pack(pady=(4, 20), padx=30, anchor="e")

    def _construir_col_original(self, contenedor):
        """Columna de imagen original con el botón Cargar Imagen."""
        col = ctk.CTkFrame(contenedor, fg_color=C_TARJETA,
                           corner_radius=10,
                           border_color=C_BORDE, border_width=1)
        col.pack(side="left", padx=5, pady=6, anchor="n")

        ctk.CTkLabel(col, text="Imagen Original",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TITULO).pack(pady=(10, 4), padx=10)

        self._lbl_original = self._cuadro_imagen(
            col, ANCHO_ORIG, ALTO_ORIG, "Carga\nuna imagen"
        )

        ctk.CTkButton(
            col,
            text="📂   Cargar Imagen",
            fg_color=C_AZUL, hover_color=C_AZUL_HOVER,
            height=36,
            command=self._cargar_imagen
        ).pack(pady=(10, 14), padx=12)

    def _construir_col_canal(self, contenedor, letra):
        """
        Columna de un canal de color (R, G o B).
        Estructura vertical:
            título del canal
            → imagen en escala de grises (actualizada en tiempo real)
            → curva interactiva (con histograma original de fondo)
            → etiqueta Min / Max
            → botón Limpiar canal
        """
        color = COLOR_CANAL[letra]
        col   = ctk.CTkFrame(contenedor, fg_color=C_TARJETA,
                              corner_radius=10,
                              border_color=C_BORDE, border_width=1)
        col.pack(side="left", padx=5, pady=6, anchor="n")

        # Título del canal
        ctk.CTkLabel(col, text=NOMBRE_CANAL[letra],
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=color).pack(pady=(10, 4), padx=10)

        # Imagen en escala de grises (se actualiza cuando el usuario mueve la curva)
        lbl_img = self._cuadro_imagen(col, ANCHO_CANAL, ALTO_CANAL, "—")
        self._lbl_canal_top[letra] = lbl_img

        # Curva interactiva (el histograma original está de fondo semitransparente)
        curva = CurvaCanal(
            col,
            color_canal=color,
            callback=lambda vmin, vmax, l=letra: self._al_cambiar_curva(l, vmin, vmax),
            ancho_pulgadas=3.5,
            alto_pulgadas=2.6,
        )
        curva.pack(pady=(4, 2), padx=8)
        self._curvas[letra] = curva

        # Etiqueta con los valores actuales de min y max
        lbl_mm = ctk.CTkLabel(
            col,
            text="Min: 0   Max: 255",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=C_SUBTITULO
        )
        lbl_mm.pack(pady=2)
        self._lbl_minmax[letra] = lbl_mm

        # Botón para resetear solo este canal
        ctk.CTkButton(
            col,
            text=f"↺  Limpiar {letra}",
            fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
            height=28, width=110,
            font=ctk.CTkFont(size=11),
            command=lambda l=letra: self._reset_canal(l)
        ).pack(pady=(2, 12))

    def _construir_col_normalizado(self, contenedor, letra):
        """
        Columna de análisis del canal ya normalizado.
        Estructura vertical:
            título
            → imagen del canal normalizado
            → histograma con eje Y FIJO (para ver la redistribución)
        """
        color = COLOR_CANAL[letra]
        col   = ctk.CTkFrame(contenedor, fg_color=C_TARJETA,
                              corner_radius=10,
                              border_color=C_BORDE, border_width=1)
        col.pack(side="left", padx=8, pady=4)

        ctk.CTkLabel(
            col,
            text=f"{NOMBRE_CANAL[letra]} — Normalizado",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=color
        ).pack(pady=(8, 4), padx=10)

        # Imagen del canal normalizado (encima del histograma)
        lbl = self._cuadro_imagen(col, ANCHO_NORM, ALTO_NORM, "—")
        self._lbl_canal_norm[letra] = lbl

        # Histograma normalizado con eje Y fijo
        # El ancho del histograma coincide con el ancho de la imagen (ANCHO_NORM)
        ancho_fig = ANCHO_NORM / 80   # pulgadas, dpi=80 → pixeles = ANCHO_NORM
        alto_fig  = 2.0               # 2 pulgadas × 80 dpi = 160 px de alto

        fig = Figure(figsize=(ancho_fig, alto_fig), dpi=80, facecolor=C_DARK_AX)
        ax  = fig.add_subplot(111)
        self._estilizar_eje_hist(ax)
        fig.tight_layout(pad=0.3)

        cv = FigureCanvasTkAgg(fig, master=col)
        cv.get_tk_widget().pack(pady=(4, 12), padx=8)

        self._ax_hist_norm[letra] = ax
        self._cv_hist_norm[letra] = cv

    # ═══════════════════════════════════════════════════════════════════════
    # PÁGINA 2 — Compresión y Binarización
    # ═══════════════════════════════════════════════════════════════════════

    def _construir_pagina2(self):
        p = self._pagina2

        ctk.CTkLabel(
            p,
            text="Paso 2 de 2 — Compresión por Bloques y Binarización",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=C_TITULO
        ).pack(pady=(16, 2))

        ctk.CTkLabel(
            p,
            text=(
                "La compresión y el threshold se aplican sobre la imagen ya normalizada "
                "del paso anterior. Los cambios se reflejan en tiempo real."
            ),
            font=ctk.CTkFont(size=11),
            text_color=C_SUBTITULO
        ).pack(pady=(0, 14))

        # ── Fila de tres imágenes ─────────────────────────────────────────
        fila_imgs = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_imgs.pack(pady=4, padx=12)

        # Imagen 1: Original normalizada (referencia del paso anterior)
        col1 = ctk.CTkFrame(fila_imgs, fg_color=C_TARJETA,
                            corner_radius=10,
                            border_color=C_BORDE, border_width=1)
        col1.pack(side="left", padx=6, pady=4)
        ctk.CTkLabel(col1, text="Original Normalizada",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TITULO).pack(pady=(10, 4))
        self._lbl_p2_orig = self._cuadro_imagen(col1, ANCHO_P2, ALTO_P2, "—")
        self._lbl_dim_orig = ctk.CTkLabel(
            col1, text="", font=ctk.CTkFont(size=10), text_color=C_SUBTITULO)
        self._lbl_dim_orig.pack(pady=(2, 10))

        # Imagen 2: Comprimida por bloques (en escala de grises)
        col2 = ctk.CTkFrame(fila_imgs, fg_color=C_TARJETA,
                            corner_radius=10,
                            border_color=C_BORDE, border_width=1)
        col2.pack(side="left", padx=6, pady=4)
        ctk.CTkLabel(col2, text="Imagen Comprimida (escala de grises)",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TITULO).pack(pady=(10, 4))
        self._lbl_p2_comp = self._cuadro_imagen(col2, ANCHO_P2, ALTO_P2, "—")
        self._lbl_dim_comp = ctk.CTkLabel(
            col2, text="", font=ctk.CTkFont(size=10), text_color=C_SUBTITULO)
        self._lbl_dim_comp.pack(pady=(2, 10))

        # Imagen 3: Binaria (solo negro y blanco)
        col3 = ctk.CTkFrame(fila_imgs, fg_color=C_TARJETA,
                            corner_radius=10,
                            border_color=C_BORDE, border_width=1)
        col3.pack(side="left", padx=6, pady=4)
        ctk.CTkLabel(col3, text="Imagen Binaria (Threshold)",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C_TITULO).pack(pady=(10, 4))
        self._lbl_p2_bin = self._cuadro_imagen(col3, ANCHO_P2, ALTO_P2, "—")
        self._lbl_info_bin = ctk.CTkLabel(
            col3, text="", font=ctk.CTkFont(size=10), text_color=C_SUBTITULO)
        self._lbl_info_bin.pack(pady=(2, 10))

        # ── Panel de controles ────────────────────────────────────────────
        panel = ctk.CTkFrame(p, fg_color=C_TARJETA,
                              corner_radius=10,
                              border_color=C_BORDE, border_width=1)
        panel.pack(fill="x", padx=20, pady=14)

        ctk.CTkLabel(panel, text="Controles de Procesamiento",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C_TITULO).pack(anchor="w", padx=16, pady=(12, 6))

        ctk.CTkFrame(panel, height=1, fg_color=C_BORDE).pack(fill="x", padx=16)

        # ─ Tamaño de bloque ───────────────────────────────────────────────
        sec_bloque = ctk.CTkFrame(panel, fg_color=C_TARJETA)
        sec_bloque.pack(anchor="w", padx=16, pady=(12, 4))

        ctk.CTkLabel(
            sec_bloque,
            text="Tamaño de bloque para compresión por promedio:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_TITULO
        ).pack(anchor="w", pady=(0, 6))

        fila_radio = ctk.CTkFrame(sec_bloque, fg_color=C_TARJETA)
        fila_radio.pack(anchor="w")

        self._var_bloque = ctk.IntVar(value=2)
        for tam in (2, 4, 8, 16):
            ctk.CTkRadioButton(
                fila_radio,
                text=f"   {tam}×{tam}   ",
                variable=self._var_bloque,
                value=tam,
                command=self._al_cambiar_bloque,
                text_color=C_TITULO,
                radiobutton_width=16,
                radiobutton_height=16,
                font=ctk.CTkFont(size=12)
            ).pack(side="left", padx=14)

        ctk.CTkLabel(
            sec_bloque,
            text="Cada bloque de N×N píxeles se reemplaza por el promedio de sus valores.",
            font=ctk.CTkFont(size=10),
            text_color=C_SUBTITULO
        ).pack(anchor="w", pady=(6, 0))

        ctk.CTkFrame(panel, height=1, fg_color=C_BORDE).pack(fill="x", padx=16, pady=10)

        # ─ Umbral de binarización ─────────────────────────────────────────
        sec_umbral = ctk.CTkFrame(panel, fg_color=C_TARJETA)
        sec_umbral.pack(fill="x", padx=16, pady=(0, 12))

        # Fila con el título + media calculada (referencia)
        fila_tit = ctk.CTkFrame(sec_umbral, fg_color=C_TARJETA)
        fila_tit.pack(fill="x")

        ctk.CTkLabel(
            fila_tit,
            text="Umbral de binarización (threshold):",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_TITULO
        ).pack(side="left", pady=(0, 6))

        self._lbl_media_ref = ctk.CTkLabel(
            fila_tit,
            text="   |   Media de la imagen: —",
            font=ctk.CTkFont(size=11),
            text_color=C_SUBTITULO
        )
        self._lbl_media_ref.pack(side="left", pady=(0, 6))

        # Fila con el slider + valor numérico
        fila_slider = ctk.CTkFrame(sec_umbral, fg_color=C_TARJETA)
        fila_slider.pack(fill="x")

        self._slider_umbral = ctk.CTkSlider(
            fila_slider,
            from_=0, to=255, number_of_steps=255,
            command=self._al_cambiar_umbral,
            width=400
        )
        self._slider_umbral.set(128)
        self._slider_umbral.pack(side="left", padx=(0, 14))

        self._lbl_umbral_val = ctk.CTkLabel(
            fila_slider,
            text="Umbral: 128",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color=C_TITULO,
            width=110
        )
        self._lbl_umbral_val.pack(side="left")

        # Explicación de la regla de threshold
        ctk.CTkLabel(
            sec_umbral,
            text=(
                "pixel ≥ umbral  →  255 (blanco)     |     "
                "pixel < umbral  →  0 (negro)"
            ),
            font=ctk.CTkFont(size=10),
            text_color=C_SUBTITULO
        ).pack(anchor="w", pady=(6, 0))

        # ── Botones de navegación ─────────────────────────────────────────
        fila_nav = ctk.CTkFrame(p, fg_color=C_FONDO)
        fila_nav.pack(fill="x", padx=20, pady=(8, 20))

        ctk.CTkButton(
            fila_nav,
            text="←   Volver a Normalización",
            fg_color=C_GRIS_BTN, hover_color=C_GRIS_HOVER,
            height=36,
            command=self._ir_pagina1
        ).pack(side="left")

        ctk.CTkButton(
            fila_nav,
            text="💾   Guardar Imagen Binaria",
            fg_color=C_VERDE, hover_color=C_VERDE_HOVER,
            height=36,
            command=self._guardar_binaria
        ).pack(side="right")

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Carga de imagen
    # ═══════════════════════════════════════════════════════════════════════

    def _cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not ruta:
            return

        img = cv2.imread(ruta)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen seleccionada.")
            return

        self.imagen_original = img

        # Separar los tres canales de color (devuelve R, G, B por separado)
        r, g, b = separar_canales(img)
        self.canales_orig = {"R": r, "G": g, "B": b}

        # Calcular histogramas originales de los tres canales
        histogramas_orig = {l: calcular_histograma(self.canales_orig[l]) for l in "RGB"}

        # Determinar el máximo de conteo para fijar el eje Y de todos los histogramas.
        # Usar el mismo techo en todos los histogramas permite comparar la distribución
        # antes y después de normalizar sin que la escala engañe al observador.
        self.y_max_hist = max(h.max() for h in histogramas_orig.values())
        if self.y_max_hist < 1:
            self.y_max_hist = 1

        # Cargar el histograma en cada curva (sin disparar el callback todavía)
        # y resetear los valores a min=0, max=255 (sin cambio visual)
        for letra in "RGB":
            self._curvas[letra].cargar_histograma(histogramas_orig[letra])
            self._curvas[letra].set_valores(0, 255)
            self._lbl_minmax[letra].configure(text="Min: 0   Max: 255")

        # Mostrar la imagen original en la sección de Imagen Original
        self._mostrar(self._lbl_original, img, "orig", ANCHO_ORIG, ALTO_ORIG)

        # Inicializar canales normalizados como copias exactas de los originales
        # (min=0 y max=255 es la identidad: sin ningún cambio)
        for letra in "RGB":
            self.canales_norm[letra] = self.canales_orig[letra].copy()
            self._mostrar(self._lbl_canal_top[letra],
                          self.canales_norm[letra],
                          f"canal_top_{letra}", ANCHO_CANAL, ALTO_CANAL)
            self._mostrar(self._lbl_canal_norm[letra],
                          self.canales_norm[letra],
                          f"canal_norm_{letra}", ANCHO_NORM, ALTO_NORM)
            hist = calcular_histograma(self.canales_norm[letra])
            self._dibujar_histograma_normalizado(letra, hist)

        # Mostrar la imagen recombinada (al inicio = igual a la original)
        self._actualizar_recombinada()

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Normalización en tiempo real
    # ═══════════════════════════════════════════════════════════════════════

    def _al_cambiar_curva(self, letra, vmin, vmax):
        """
        Callback que se ejecuta cada vez que el usuario arrastra un punto
        en la curva de un canal. Actualiza todas las visualizaciones en tiempo real.
        """
        self._lbl_minmax[letra].configure(text=f"Min: {vmin}   Max: {vmax}")

        # Si aún no hay imagen cargada, no hacemos nada más
        if self.imagen_original is None:
            return

        self._normalizar_y_mostrar_canal(letra, vmin, vmax)
        self._actualizar_recombinada()

    def _normalizar_y_mostrar_canal(self, letra, vmin, vmax):
        """
        Aplica la normalización min-max al canal indicado y actualiza
        su imagen y su histograma en ambas secciones de la página 1.
        """
        # Fórmula:
        #   pixel < vmin  → 0
        #   pixel > vmax  → 255
        #   entre vmin y vmax: (pixel - vmin) / (vmax - vmin) × 255
        canal_normalizado = normalizar_canal(self.canales_orig[letra], vmin, vmax)
        self.canales_norm[letra] = canal_normalizado

        # Actualizar imagen sobre la curva (parte superior, en tiempo real)
        self._mostrar(self._lbl_canal_top[letra],
                      canal_normalizado,
                      f"canal_top_{letra}", ANCHO_CANAL, ALTO_CANAL)

        # Actualizar imagen en la sección inferior (análisis de canales normalizados)
        self._mostrar(self._lbl_canal_norm[letra],
                      canal_normalizado,
                      f"canal_norm_{letra}", ANCHO_NORM, ALTO_NORM)

        # Actualizar histograma normalizado (con el eje Y fijo)
        hist = calcular_histograma(canal_normalizado)
        self._dibujar_histograma_normalizado(letra, hist)

    def _actualizar_recombinada(self):
        """
        Une los tres canales normalizados y muestra la imagen recombinada.
        Solo se ejecuta cuando los tres canales están disponibles.
        """
        if any(v is None for v in self.canales_norm.values()):
            return

        self.imagen_recombinada = unir_canales(
            self.canales_norm["R"],
            self.canales_norm["G"],
            self.canales_norm["B"]
        )
        self._mostrar(self._lbl_recombinada, self.imagen_recombinada,
                      "recomb", ANCHO_ORIG * 2, ALTO_ORIG)

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Botones de reset
    # ═══════════════════════════════════════════════════════════════════════

    def _reset_canal(self, letra):
        """Restaura min=0 y max=255 en el canal indicado (solo ese canal)."""
        self._curvas[letra].set_valores(0, 255)
        self._lbl_minmax[letra].configure(text="Min: 0   Max: 255")

        if self.imagen_original is not None:
            self._normalizar_y_mostrar_canal(letra, 0, 255)
            self._actualizar_recombinada()

    def _reset_todo(self):
        """Restaura los tres canales a sin normalización (min=0, max=255)."""
        for letra in "RGB":
            self._reset_canal(letra)

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Página 2 — Compresión y Threshold
    # ═══════════════════════════════════════════════════════════════════════

    def _actualizar_pagina2(self):
        """Se llama al navegar a la página 2. Inicializa las tres imágenes."""
        if self.imagen_recombinada is None:
            return

        # Imagen de referencia: la ya normalizada del paso anterior
        h, w = self.imagen_recombinada.shape[:2]
        self._mostrar(self._lbl_p2_orig, self.imagen_recombinada,
                      "p2_orig", ANCHO_P2, ALTO_P2)
        self._lbl_dim_orig.configure(text=f"Dimensiones: {w} × {h} px")

        # Aplicar compresión y threshold con los valores actuales del slider
        self._aplicar_compresion_y_threshold()

    def _al_cambiar_bloque(self):
        """Se llama cuando el usuario selecciona un tamaño de bloque diferente."""
        self.bloque_actual = self._var_bloque.get()
        if self.imagen_recombinada is not None:
            self._aplicar_compresion_y_threshold()

    def _al_cambiar_umbral(self, valor):
        """Se llama cuando el usuario mueve el slider de umbral."""
        self.umbral_actual = int(float(valor))
        self._lbl_umbral_val.configure(text=f"Umbral: {self.umbral_actual}")

        # Solo re-aplicamos el threshold (la compresión no cambia)
        if self.imagen_comprimida is not None:
            self._aplicar_threshold_solo()

    def _aplicar_compresion_y_threshold(self):
        """
        Convierte la imagen normalizada a escala de grises,
        aplica la compresión por bloques y luego el threshold.
        """
        # Paso 1: convertir a escala de grises usando fórmula de luminancia
        #         0.299 × R + 0.587 × G + 0.114 × B
        gris = convertir_a_grises(self.imagen_recombinada)

        # Paso 2: compresión por promedio de bloques NxN
        comp = comprimir_por_bloques(gris, self.bloque_actual)
        self.imagen_comprimida = comp

        h, w = comp.shape[:2]
        self._mostrar(self._lbl_p2_comp, comp, "p2_comp", ANCHO_P2, ALTO_P2)
        self._lbl_dim_comp.configure(
            text=f"Bloque {self.bloque_actual}×{self.bloque_actual}   |   {w}×{h} px"
        )

        # Paso 3: binarización con el umbral actual
        self._aplicar_threshold_solo()

    def _aplicar_threshold_solo(self):
        """
        Aplica solo el threshold sobre la imagen comprimida actual,
        sin recalcular la compresión.
        """
        if self.imagen_comprimida is None:
            return

        # Si el umbral es la media, pasamos None; si es un valor fijo, lo pasamos
        binaria, media, umbral_usado = aplicar_threshold(
            self.imagen_comprimida, self.umbral_actual
        )
        self.imagen_binaria = binaria

        self._mostrar(self._lbl_p2_bin, binaria, "p2_bin", ANCHO_P2, ALTO_P2)
        self._lbl_info_bin.configure(
            text=f"Umbral aplicado: {umbral_usado}   |   Media de la imagen: {media:.1f}"
        )
        self._lbl_media_ref.configure(
            text=f"   |   Media de la imagen: {media:.1f}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # LÓGICA: Guardar imagen final
    # ═══════════════════════════════════════════════════════════════════════

    def _guardar_binaria(self):
        if self.imagen_binaria is None:
            messagebox.showwarning(
                "Sin imagen",
                "Genera primero la imagen binaria navegando a esta página\n"
                "con una imagen ya normalizada."
            )
            return

        os.makedirs("imagenes", exist_ok=True)

        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen binaria",
            initialdir="imagenes",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("Todos", "*.*")
            ]
        )
        if ruta:
            cv2.imwrite(ruta, self.imagen_binaria)
            messagebox.showinfo("Guardado", f"Imagen binaria guardada en:\n{ruta}")

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS DE INTERFAZ
    # ═══════════════════════════════════════════════════════════════════════

    def _cuadro_imagen(self, contenedor, ancho, alto, texto=""):
        """
        Crea un frame de tamaño fijo con un Label para mostrar imágenes.
        pack_propagate(False) garantiza que el frame NO se encoja aunque
        la imagen sea más pequeña o aún no esté cargada.
        Devuelve el Label para que el llamador pueda actualizarlo con .config().
        """
        marco = ctk.CTkFrame(
            contenedor, width=ancho, height=alto,
            fg_color=C_PLACEHOLDER,
            corner_radius=4,
            border_color=C_BORDE, border_width=1
        )
        marco.pack(pady=4, padx=8)
        marco.pack_propagate(False)

        lbl = tk.Label(
            marco, text=texto,
            bg=C_PLACEHOLDER, fg=C_SUBTITULO,
            font=("Arial", 9, "italic"),
            wraplength=ancho - 16,
            anchor="center"
        )
        lbl.pack(expand=True)
        return lbl

    def _mostrar(self, label, imagen_numpy, clave, ancho_max, alto_max):
        """
        Convierte una imagen OpenCV (numpy) a PhotoImage de tkinter
        y la asigna al Label. Guarda la referencia en self.refs para
        evitar que el recolector de basura la elimine.
        """
        if imagen_numpy is None:
            return
        img_tk = convertir_imagen_para_tkinter(imagen_numpy, ancho_max, alto_max)
        if img_tk is None:
            return
        self.refs[clave] = img_tk
        label.config(image=img_tk, text="")

    def _estilizar_eje_hist(self, ax):
        """Aplica el estilo oscuro a un eje de matplotlib para histogramas."""
        ax.set_facecolor(C_DARK_AX)
        ax.tick_params(labelsize=6, colors=C_TICK)
        for sp in ax.spines.values():
            sp.set_color(C_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 255)

    def _dibujar_histograma_normalizado(self, letra, frecuencias):
        """
        Dibuja el histograma del canal normalizado con eje Y FIJO.
        El eje Y siempre va de 0 al máximo de los histogramas originales,
        lo que permite observar visualmente cómo la normalización
        redistribuye (estira o corta) la distribución de píxeles.
        """
        ax = self._ax_hist_norm[letra]
        ax.clear()
        self._estilizar_eje_hist(ax)

        # Eje Y fijo: usamos el máximo de los histogramas originales × 1.05
        ax.set_ylim(0, self.y_max_hist * 1.05)

        ax.bar(range(256), frecuencias,
               color=COLOR_CANAL[letra], width=1.0, alpha=0.85)
        ax.grid(True, alpha=0.12, color="white", linewidth=0.5)

        self._cv_hist_norm[letra].draw_idle()
