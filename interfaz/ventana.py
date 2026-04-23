# ============================================================
# Módulo: ventana.py
# ============================================================
# Clase principal de la interfaz. Organiza la ventana en dos páginas
# con navegación por migas de pan (breadcrumb):
#
#   PÁGINA 1 - Normalización:
#     Fila superior: Imagen original | Canal R | Canal G | Canal B
#                    (imagen en escala de grises + histograma + slider por canal)
#     Fila inferior: Imagen recombinada | R' | G' | B'
#                    (imagen normalizada + histograma por canal)
#
#   PÁGINA 2 - Compresión y Threshold:
#     Imagen original arriba (previsualización)
#     Selector de tamaño de bloque
#     Imagen comprimida (resultado)
#     Imagen binaria (threshold)

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Módulos de procesamiento
from procesamiento.canales import separar_canales, unir_canales
from procesamiento.histograma import calcular_histograma
from procesamiento.redistribuir import normalizar_canal
from procesamiento.compresion import convertir_a_grises, comprimir_por_bloques
from procesamiento.threshold import aplicar_threshold

# Componentes de la interfaz
from interfaz.componentes import (
    convertir_imagen_para_tkinter,
    crear_figura_histograma,
    dibujar_histograma,
    MarcoScrollable,
    RangeSlider,
)


# ============================================================
# Paleta de colores
# ============================================================
COLOR_FONDO = "#f0f2f5"
COLOR_TARJETA = "#ffffff"
COLOR_PLACEHOLDER = "#f7f8fa"
COLOR_BORDE = "#dde1e7"
COLOR_TITULO = "#1a2332"
COLOR_SUBTITULO = "#4a5568"
COLOR_TEXTO_GRIS = "#8a9ab0"

COLOR_BOTON_CARGAR = "#2563eb"
COLOR_BOTON_RESET = "#dc7420"
COLOR_BOTON_GUARDAR = "#16a34a"

COLOR_NAV_ACTIVO = "#2563eb"
COLOR_NAV_INACTIVO = COLOR_TARJETA

COLOR_CANAL_ROJO = "#dc2626"
COLOR_CANAL_VERDE = "#16a34a"
COLOR_CANAL_AZUL = "#2563eb"

# Tamaños de imagen para los contenedores fijos
ANCHO_IMG_GRANDE = 340
ALTO_IMG_GRANDE = 260

ANCHO_IMG_CANAL = 270
ALTO_IMG_CANAL = 195

ANCHO_IMG_RESULTADO = 480
ALTO_IMG_RESULTADO = 350


# ============================================================
# Clase principal
# ============================================================

class VentanaPrincipal:

    def __init__(self, ventana_raiz):
        self.ventana_raiz = ventana_raiz
        self.ventana_raiz.configure(bg=COLOR_FONDO)

        # --- Imágenes en cada etapa ---
        self.imagen_original = None
        self.canal_rojo_original = None
        self.canal_verde_original = None
        self.canal_azul_original = None
        self.canal_rojo_normalizado = None
        self.canal_verde_normalizado = None
        self.canal_azul_normalizado = None
        self.imagen_recombinada = None
        self.imagen_en_grises = None
        self.imagen_comprimida = None
        self.imagen_binaria = None

        # Guardamos las referencias a PhotoImage para que Python no las elimine
        self.referencias_imagenes_tk = {}

        self.tamano_bloque_seleccionado = tk.IntVar(value=4)

        self._construir_interfaz()

    # ================================================================
    # CONSTRUCCIÓN GENERAL
    # ================================================================

    def _construir_interfaz(self):
        # Encabezado fijo (no hace scroll)
        self._construir_encabezado()

        # Área de páginas (ocupa el resto, hace scroll internamente)
        self.contenedor_paginas = tk.Frame(self.ventana_raiz, bg=COLOR_FONDO)
        self.contenedor_paginas.pack(fill="both", expand=True)

        # Construimos cada página y las guardamos en MarcoScrollable
        self.pagina_normalizacion = MarcoScrollable(
            self.contenedor_paginas, color_fondo=COLOR_FONDO
        )
        self.pagina_compresion = MarcoScrollable(
            self.contenedor_paginas, color_fondo=COLOR_FONDO
        )

        self._construir_pagina_normalizacion(
            self.pagina_normalizacion.contenido_interno
        )
        self._construir_pagina_compresion(
            self.pagina_compresion.contenido_interno
        )

        # Mostramos la página 1 por defecto
        self._ir_a_normalizacion()

    # ================================================================
    # ENCABEZADO (título + botones + breadcrumb)
    # ================================================================

    def _construir_encabezado(self):
        marco_encabezado = tk.Frame(
            self.ventana_raiz, bg=COLOR_TARJETA, pady=12,
            highlightthickness=1, highlightbackground=COLOR_BORDE
        )
        marco_encabezado.pack(fill="x")

        # Título
        tk.Label(
            marco_encabezado,
            text="Ecualizador de Imágenes — Preprocesamiento ML",
            font=("Arial", 16, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack()

        # Botonera de acciones
        marco_botones = tk.Frame(marco_encabezado, bg=COLOR_TARJETA)
        marco_botones.pack(pady=(8, 6))

        self._crear_boton(
            marco_botones, "Cargar Imagen", COLOR_BOTON_CARGAR, self.cargar_imagen
        ).pack(side="left", padx=5)

        self._crear_boton(
            marco_botones, "Reset", COLOR_BOTON_RESET, self.resetear_aplicacion
        ).pack(side="left", padx=5)

        self._crear_boton(
            marco_botones, "Guardar Imagen Final", COLOR_BOTON_GUARDAR, self.guardar_imagen_final
        ).pack(side="left", padx=5)

        # Separador
        tk.Frame(marco_encabezado, height=1, bg=COLOR_BORDE).pack(fill="x", pady=(6, 0))

        # Breadcrumb / migas de pan
        marco_breadcrumb = tk.Frame(marco_encabezado, bg=COLOR_TARJETA)
        marco_breadcrumb.pack(pady=(6, 0))

        self.boton_nav_normalizacion = tk.Button(
            marco_breadcrumb,
            text="1. Normalización de Canales",
            font=("Arial", 10, "bold"),
            bd=0, padx=18, pady=6, cursor="hand2",
            command=self._ir_a_normalizacion
        )
        self.boton_nav_normalizacion.pack(side="left", padx=2)

        tk.Label(
            marco_breadcrumb, text="›",
            font=("Arial", 14), bg=COLOR_TARJETA, fg=COLOR_TEXTO_GRIS
        ).pack(side="left")

        self.boton_nav_compresion = tk.Button(
            marco_breadcrumb,
            text="2. Compresión y Threshold",
            font=("Arial", 10, "bold"),
            bd=0, padx=18, pady=6, cursor="hand2",
            command=self._ir_a_compresion
        )
        self.boton_nav_compresion.pack(side="left", padx=2)

    def _crear_boton(self, contenedor, texto, color, comando):
        return tk.Button(
            contenedor, text=texto, command=comando,
            bg=color, fg="white",
            font=("Arial", 10, "bold"),
            activebackground=color, activeforeground="white",
            bd=0, padx=20, pady=8, cursor="hand2"
        )

    # ================================================================
    # NAVEGACIÓN ENTRE PÁGINAS (breadcrumb)
    # ================================================================

    def _ir_a_normalizacion(self):
        self.pagina_compresion.pack_forget()
        self.pagina_normalizacion.pack(fill="both", expand=True)
        self.boton_nav_normalizacion.config(bg=COLOR_NAV_ACTIVO, fg="white")
        self.boton_nav_compresion.config(bg=COLOR_NAV_INACTIVO, fg=COLOR_SUBTITULO)

    def _ir_a_compresion(self):
        self.pagina_normalizacion.pack_forget()
        self.pagina_compresion.pack(fill="both", expand=True)
        self.boton_nav_normalizacion.config(bg=COLOR_NAV_INACTIVO, fg=COLOR_SUBTITULO)
        self.boton_nav_compresion.config(bg=COLOR_NAV_ACTIVO, fg="white")
        # Sincronizamos la previsualización de la imagen original en página 2
        self._actualizar_original_pagina2()

    # ================================================================
    # PÁGINA 1 — NORMALIZACIÓN
    # ================================================================

    def _construir_pagina_normalizacion(self, contenedor):

        # ---- SECCIÓN SUPERIOR: Imagen original + 3 canales originales ----
        seccion_original = self._crear_tarjeta(
            contenedor,
            "Imagen Original y Canales  |  mueve los sliders para normalizar"
        )

        fila_superior = tk.Frame(seccion_original, bg=COLOR_TARJETA)
        fila_superior.pack(fill="x")

        # Columna 0 — imagen original grande
        marco_col_original = tk.Frame(fila_superior, bg=COLOR_TARJETA)
        marco_col_original.pack(side="left", padx=15, anchor="n", pady=5)

        tk.Label(
            marco_col_original, text="Imagen Original",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(pady=(0, 4))

        self.label_imagen_original = self._crear_contenedor_imagen(
            marco_col_original,
            ANCHO_IMG_GRANDE, ALTO_IMG_GRANDE,
            "Carga una imagen\npara comenzar"
        )

        # Columnas 1-3 — canales R, G, B originales
        self.widgets_canal_rojo = self._construir_col_canal_original(
            fila_superior, "Canal Rojo (R)", COLOR_CANAL_ROJO, indice_canal=0
        )
        self.widgets_canal_verde = self._construir_col_canal_original(
            fila_superior, "Canal Verde (G)", COLOR_CANAL_VERDE, indice_canal=1
        )
        self.widgets_canal_azul = self._construir_col_canal_original(
            fila_superior, "Canal Azul (B)", COLOR_CANAL_AZUL, indice_canal=2
        )

        # ---- SECCIÓN INFERIOR: Imagen recombinada + 3 canales normalizados ----
        seccion_recombinada = self._crear_tarjeta(
            contenedor,
            "Imagen Recombinada y Canales Normalizados"
        )

        fila_inferior = tk.Frame(seccion_recombinada, bg=COLOR_TARJETA)
        fila_inferior.pack(fill="x")

        # Columna 0 — imagen recombinada grande
        marco_col_recombinada = tk.Frame(fila_inferior, bg=COLOR_TARJETA)
        marco_col_recombinada.pack(side="left", padx=15, anchor="n", pady=5)

        tk.Label(
            marco_col_recombinada, text="Imagen Recombinada",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(pady=(0, 4))

        self.label_imagen_recombinada = self._crear_contenedor_imagen(
            marco_col_recombinada,
            ANCHO_IMG_GRANDE, ALTO_IMG_GRANDE,
            "Esperando\nnormalización"
        )

        # Columnas 1-3 — canales normalizados
        self.widgets_canal_rojo_mod = self._construir_col_canal_modificado(
            fila_inferior, "R' Normalizado", COLOR_CANAL_ROJO
        )
        self.widgets_canal_verde_mod = self._construir_col_canal_modificado(
            fila_inferior, "G' Normalizado", COLOR_CANAL_VERDE
        )
        self.widgets_canal_azul_mod = self._construir_col_canal_modificado(
            fila_inferior, "B' Normalizado", COLOR_CANAL_AZUL
        )

    def _construir_col_canal_original(self, contenedor_fila, titulo, color, indice_canal):
        """
        Columna de canal ORIGINAL:  imagen en grises → histograma → range slider.
        """
        marco = tk.Frame(contenedor_fila, bg=COLOR_TARJETA)
        marco.pack(side="left", padx=10, anchor="n", pady=5)

        tk.Label(
            marco, text=titulo,
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=color
        ).pack(pady=(0, 4))

        # Imagen del canal en escala de grises (arriba del histograma)
        label_imagen = self._crear_contenedor_imagen(
            marco, ANCHO_IMG_CANAL, ALTO_IMG_CANAL, "sin cargar"
        )

        # Histograma debajo de la imagen
        figura, eje = crear_figura_histograma()
        canvas_histograma = FigureCanvasTkAgg(figura, master=marco)
        canvas_histograma.get_tk_widget().pack(pady=(4, 2))

        # RangeSlider (un solo slider con dos manejadores: min y max)
        slider_rango = RangeSlider(
            marco,
            desde=0, hasta=255,
            valor_min_inicial=0, valor_max_inicial=255,
            ancho=ANCHO_IMG_CANAL,
            color_activo=color,
            color_fondo=COLOR_TARJETA,
            callback=lambda vmin, vmax, idx=indice_canal:
                self.al_cambiar_slider(idx, vmin, vmax)
        )
        slider_rango.pack(pady=(2, 5))

        return {
            "label_imagen": label_imagen,
            "figura": figura,
            "eje": eje,
            "canvas_histograma": canvas_histograma,
            "slider_rango": slider_rango,
            "color": color,
        }

    def _construir_col_canal_modificado(self, contenedor_fila, titulo, color):
        """
        Columna de canal MODIFICADO/NORMALIZADO:  imagen en grises → histograma.
        (Sin slider, porque el slider está en la sección de originales.)
        """
        marco = tk.Frame(contenedor_fila, bg=COLOR_TARJETA)
        marco.pack(side="left", padx=10, anchor="n", pady=5)

        tk.Label(
            marco, text=titulo,
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=color
        ).pack(pady=(0, 4))

        # Imagen del canal normalizado (arriba del histograma)
        label_imagen = self._crear_contenedor_imagen(
            marco, ANCHO_IMG_CANAL, ALTO_IMG_CANAL, "sin procesar"
        )

        # Histograma debajo
        figura, eje = crear_figura_histograma()
        canvas_histograma = FigureCanvasTkAgg(figura, master=marco)
        canvas_histograma.get_tk_widget().pack(pady=(4, 5))

        return {
            "label_imagen": label_imagen,
            "figura": figura,
            "eje": eje,
            "canvas_histograma": canvas_histograma,
            "color": color,
        }

    # ================================================================
    # PÁGINA 2 — COMPRESIÓN Y THRESHOLD
    # ================================================================

    def _construir_pagina_compresion(self, contenedor):

        # ---- Imagen original (previsualización de referencia) ----
        seccion_ref = self._crear_tarjeta(
            contenedor,
            "Imagen de referencia (normalizada)"
        )

        marco_ref = tk.Frame(seccion_ref, bg=COLOR_TARJETA)
        marco_ref.pack(anchor="center")

        tk.Label(
            marco_ref, text="Imagen tras normalización",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(pady=(0, 4))

        self.label_imagen_original_pag2 = self._crear_contenedor_imagen(
            marco_ref,
            ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO,
            "Carga una imagen en la página anterior"
        )

        # ---- Compresión por bloques ----
        seccion_compresion = self._crear_tarjeta(
            contenedor,
            "Compresión por Bloques (grises + promedio por bloque)"
        )

        # Selector de tamaño de bloque
        marco_selector = tk.Frame(seccion_compresion, bg=COLOR_TARJETA)
        marco_selector.pack(pady=(0, 10))

        tk.Label(
            marco_selector, text="Tamaño de bloque:",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(side="left", padx=(0, 12))

        for tam in [2, 4, 8, 16, 32]:
            tk.Radiobutton(
                marco_selector,
                text=f"{tam}×{tam}",
                variable=self.tamano_bloque_seleccionado,
                value=tam,
                command=self.aplicar_compresion_y_threshold,
                bg=COLOR_TARJETA, fg=COLOR_TITULO,
                activebackground=COLOR_TARJETA,
                selectcolor=COLOR_TARJETA,
                font=("Arial", 10),
                cursor="hand2"
            ).pack(side="left", padx=6)

        # Imagen comprimida centrada
        marco_comprimida = tk.Frame(seccion_compresion, bg=COLOR_TARJETA)
        marco_comprimida.pack(anchor="center")

        tk.Label(
            marco_comprimida, text="Imagen comprimida",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(pady=(0, 4))

        self.label_imagen_comprimida = self._crear_contenedor_imagen(
            marco_comprimida,
            ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO,
            "Aplica normalización primero"
        )

        # ---- Threshold ----
        seccion_threshold = self._crear_tarjeta(
            contenedor,
            "Threshold — Imagen Binaria Final"
        )

        self.label_valor_media = tk.Label(
            seccion_threshold,
            text="Media usada como umbral: —",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        )
        self.label_valor_media.pack(pady=(0, 8))

        marco_binaria = tk.Frame(seccion_threshold, bg=COLOR_TARJETA)
        marco_binaria.pack(anchor="center")

        tk.Label(
            marco_binaria, text="Imagen binaria (threshold)",
            font=("Arial", 10, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO
        ).pack(pady=(0, 4))

        self.label_imagen_binaria = self._crear_contenedor_imagen(
            marco_binaria,
            ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO,
            "Esperando compresión"
        )

    # ================================================================
    # HELPERS DE UI
    # ================================================================

    def _crear_tarjeta(self, contenedor, titulo):
        """Crea una tarjeta blanca con borde y título, devuelve el frame interno."""
        marco_externo = tk.Frame(contenedor, bg=COLOR_FONDO, padx=18, pady=8)
        marco_externo.pack(fill="x")

        marco_tarjeta = tk.Frame(
            marco_externo, bg=COLOR_TARJETA,
            highlightthickness=1, highlightbackground=COLOR_BORDE
        )
        marco_tarjeta.pack(fill="x")

        tk.Label(
            marco_tarjeta, text=titulo,
            font=("Arial", 11, "bold"),
            bg=COLOR_TARJETA, fg=COLOR_TITULO, anchor="w"
        ).pack(fill="x", padx=18, pady=(12, 6))

        tk.Frame(marco_tarjeta, height=1, bg=COLOR_BORDE).pack(fill="x")

        marco_contenido = tk.Frame(marco_tarjeta, bg=COLOR_TARJETA, padx=18, pady=14)
        marco_contenido.pack(fill="x")

        return marco_contenido

    def _crear_contenedor_imagen(self, contenedor_padre, ancho, alto, texto_inicial):
        """
        Crea un cuadro fijo (ancho x alto px) donde se muestra la imagen.
        Usa pack_propagate(False) para que el cuadro no cambie de tamaño
        aunque la imagen sea más pequeña (útil con imágenes panorámicas).
        Devuelve el Label donde se asignará la imagen luego.
        """
        marco = tk.Frame(
            contenedor_padre,
            width=ancho, height=alto,
            bg=COLOR_PLACEHOLDER,
            highlightthickness=1,
            highlightbackground=COLOR_BORDE
        )
        marco.pack_propagate(False)   # <- no colapsar al tamaño del hijo
        marco.pack(pady=4)            # <- EMPAQUETAMOS el marco aquí

        label = tk.Label(
            marco, text=texto_inicial,
            bg=COLOR_PLACEHOLDER, fg=COLOR_TEXTO_GRIS,
            font=("Arial", 9, "italic"), wraplength=ancho - 10
        )
        label.pack(expand=True)        # centrado dentro del marco fijo

        return label                   # devolvemos solo el Label

    # ================================================================
    # MÉTODOS DE ACCIÓN (responden a eventos del usuario)
    # ================================================================

    def cargar_imagen(self):
        """Abre diálogo de archivo y ejecuta todo el pipeline de procesamiento."""
        ruta = filedialog.askopenfilename(
            title="Seleccione una imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not ruta:
            return

        imagen_cargada = cv2.imread(ruta)
        if imagen_cargada is None:
            messagebox.showerror("Error", "No se pudo cargar el archivo. Verifica que sea una imagen válida.")
            return

        self.imagen_original = imagen_cargada

        # Separamos en canales y guardamos los originales
        canal_r, canal_g, canal_b = separar_canales(self.imagen_original)
        self.canal_rojo_original = canal_r
        self.canal_verde_original = canal_g
        self.canal_azul_original = canal_b

        # Inicialmente los canales normalizados son iguales a los originales
        self.canal_rojo_normalizado = canal_r.copy()
        self.canal_verde_normalizado = canal_g.copy()
        self.canal_azul_normalizado = canal_b.copy()

        # Reiniciamos los sliders (por si venía de una imagen anterior)
        self.widgets_canal_rojo["slider_rango"].resetear()
        self.widgets_canal_verde["slider_rango"].resetear()
        self.widgets_canal_azul["slider_rango"].resetear()

        # Mostramos imagen original
        self._mostrar_imagen(
            self.label_imagen_original, self.imagen_original,
            "original", ANCHO_IMG_GRANDE, ALTO_IMG_GRANDE
        )

        # Mostramos canales originales (imagen + histograma)
        self._actualizar_canal(self.widgets_canal_rojo, canal_r, "r_orig")
        self._actualizar_canal(self.widgets_canal_verde, canal_g, "g_orig")
        self._actualizar_canal(self.widgets_canal_azul, canal_b, "b_orig")

        # Mostramos canales modificados (mismos que originales al principio)
        self._actualizar_canal(self.widgets_canal_rojo_mod, canal_r, "r_mod")
        self._actualizar_canal(self.widgets_canal_verde_mod, canal_g, "g_mod")
        self._actualizar_canal(self.widgets_canal_azul_mod, canal_b, "b_mod")

        self.actualizar_imagen_recombinada()
        self.aplicar_compresion_y_threshold()
        self._actualizar_original_pagina2()

    def al_cambiar_slider(self, indice_canal, valor_min, valor_max):
        """
        Se ejecuta cuando el usuario mueve el RangeSlider de algún canal.
        Normaliza ese canal y propaga el cambio hacia abajo en el pipeline.
        """
        if self.imagen_original is None:
            return

        # Identificamos qué canal se está modificando según el índice
        if indice_canal == 0:
            canal_base = self.canal_rojo_original
            widgets_mod = self.widgets_canal_rojo_mod
            clave_ref = "r_mod"
        elif indice_canal == 1:
            canal_base = self.canal_verde_original
            widgets_mod = self.widgets_canal_verde_mod
            clave_ref = "g_mod"
        else:
            canal_base = self.canal_azul_original
            widgets_mod = self.widgets_canal_azul_mod
            clave_ref = "b_mod"

        if valor_max <= valor_min:
            return

        # Aplicamos la normalización min-max sobre el canal ORIGINAL
        canal_normalizado = normalizar_canal(canal_base, valor_min, valor_max)

        if indice_canal == 0:
            self.canal_rojo_normalizado = canal_normalizado
        elif indice_canal == 1:
            self.canal_verde_normalizado = canal_normalizado
        else:
            self.canal_azul_normalizado = canal_normalizado

        # Actualizamos el canal modificado (imagen + histograma)
        self._actualizar_canal(widgets_mod, canal_normalizado, clave_ref)

        # Propagamos hacia la imagen recombinada y etapas siguientes
        self.actualizar_imagen_recombinada()
        self.aplicar_compresion_y_threshold()

    def actualizar_imagen_recombinada(self):
        """Une los 3 canales normalizados y muestra la imagen recombinada."""
        if self.canal_rojo_normalizado is None:
            return

        self.imagen_recombinada = unir_canales(
            self.canal_rojo_normalizado,
            self.canal_verde_normalizado,
            self.canal_azul_normalizado
        )
        self._mostrar_imagen(
            self.label_imagen_recombinada, self.imagen_recombinada,
            "recombinada", ANCHO_IMG_GRANDE, ALTO_IMG_GRANDE
        )

    def aplicar_compresion_y_threshold(self):
        """Convierte a grises, comprime por bloques y aplica threshold."""
        if self.imagen_recombinada is None:
            return

        # Paso 1: convertir a escala de grises con fórmula de luminancia
        self.imagen_en_grises = convertir_a_grises(self.imagen_recombinada)

        # Paso 2: comprimir con el tamaño de bloque elegido
        tam_bloque = self.tamano_bloque_seleccionado.get()
        self.imagen_comprimida = comprimir_por_bloques(
            self.imagen_en_grises, tam_bloque
        )
        self._mostrar_imagen(
            self.label_imagen_comprimida, self.imagen_comprimida,
            "comprimida", ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO
        )

        # Paso 3: threshold con la media como umbral
        self.imagen_binaria, valor_media = aplicar_threshold(self.imagen_comprimida)
        self.label_valor_media.config(
            text=f"Media usada como umbral: {valor_media:.2f}"
        )
        self._mostrar_imagen(
            self.label_imagen_binaria, self.imagen_binaria,
            "binaria", ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO
        )

    def _actualizar_original_pagina2(self):
        """Sincroniza la imagen de referencia en la página 2 con la recombinada actual."""
        imagen_para_pag2 = self.imagen_recombinada if self.imagen_recombinada is not None \
                           else self.imagen_original
        if imagen_para_pag2 is not None:
            self._mostrar_imagen(
                self.label_imagen_original_pag2, imagen_para_pag2,
                "original_pag2", ANCHO_IMG_RESULTADO, ALTO_IMG_RESULTADO
            )

    def resetear_aplicacion(self):
        """Deja la aplicación en estado inicial sin imagen cargada."""
        self.imagen_original = None
        self.canal_rojo_original = None
        self.canal_verde_original = None
        self.canal_azul_original = None
        self.canal_rojo_normalizado = None
        self.canal_verde_normalizado = None
        self.canal_azul_normalizado = None
        self.imagen_recombinada = None
        self.imagen_en_grises = None
        self.imagen_comprimida = None
        self.imagen_binaria = None
        self.referencias_imagenes_tk.clear()

        # Reseteamos sliders y limpiamos canales originales
        for widgets in (self.widgets_canal_rojo,
                        self.widgets_canal_verde,
                        self.widgets_canal_azul):
            widgets["slider_rango"].resetear()
            widgets["label_imagen"].config(image="", text="sin cargar")
            widgets["eje"].clear()
            widgets["canvas_histograma"].draw()

        # Limpiamos canales modificados
        for widgets in (self.widgets_canal_rojo_mod,
                        self.widgets_canal_verde_mod,
                        self.widgets_canal_azul_mod):
            widgets["label_imagen"].config(image="", text="sin procesar")
            widgets["eje"].clear()
            widgets["canvas_histograma"].draw()

        self.label_imagen_original.config(image="", text="Carga una imagen\npara comenzar")
        self.label_imagen_recombinada.config(image="", text="Esperando\nnormalización")
        self.label_imagen_original_pag2.config(image="", text="Carga una imagen en la página anterior")
        self.label_imagen_comprimida.config(image="", text="Aplica normalización primero")
        self.label_imagen_binaria.config(image="", text="Esperando compresión")
        self.label_valor_media.config(text="Media usada como umbral: —")
        self.tamano_bloque_seleccionado.set(4)

    def guardar_imagen_final(self):
        """Guarda la imagen binaria final en la carpeta 'imagenes'."""
        if self.imagen_binaria is None:
            messagebox.showwarning("Atención", "No hay imagen final para guardar. Carga una imagen primero.")
            return

        carpeta = "imagenes"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)

        contador = 1
        while True:
            nombre = f"imagen_final_{contador}.png"
            ruta = os.path.join(carpeta, nombre)
            if not os.path.exists(ruta):
                break
            contador += 1

        cv2.imwrite(ruta, self.imagen_binaria)
        messagebox.showinfo("Guardado exitoso", f"Imagen guardada en:\n{ruta}")

    # ================================================================
    # MÉTODOS AUXILIARES
    # ================================================================

    def _mostrar_imagen(self, label, imagen_numpy, clave, ancho_max, alto_max):
        """Convierte numpy → PhotoImage y lo muestra en el Label dado."""
        if imagen_numpy is None:
            return
        imagen_tk = convertir_imagen_para_tkinter(
            imagen_numpy, ancho_maximo=ancho_max, alto_maximo=alto_max
        )
        self.referencias_imagenes_tk[clave] = imagen_tk
        label.config(image=imagen_tk, text="")

    def _actualizar_canal(self, widgets, canal, clave):
        """Actualiza la imagen en grises y el histograma de un canal."""
        # Imagen del canal (en escala de grises)
        imagen_tk = convertir_imagen_para_tkinter(
            canal, ancho_maximo=ANCHO_IMG_CANAL, alto_maximo=ALTO_IMG_CANAL
        )
        self.referencias_imagenes_tk[clave] = imagen_tk
        widgets["label_imagen"].config(image=imagen_tk, text="")

        # Histograma del canal
        frecuencias = calcular_histograma(canal)
        dibujar_histograma(widgets["eje"], frecuencias, widgets["color"])
        widgets["canvas_histograma"].draw()
