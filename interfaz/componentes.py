# ============================================================
# Módulo: componentes.py
# ============================================================
# Contiene componentes reutilizables de la interfaz:
#   - convertir_imagen_para_tkinter : numpy -> PhotoImage (con resize)
#   - crear_figura_histograma / dibujar_histograma : helpers de matplotlib
#   - MarcoScrollable : Frame con scroll vertical que se ajusta al ancho
#   - RangeSlider : slider personalizado con dos manejadores (min y max)

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from matplotlib.figure import Figure


# ============================================================
# Conversión y escalado de imágenes para tkinter
# ============================================================

def convertir_imagen_para_tkinter(imagen_opencv, ancho_maximo=360, alto_maximo=280):
    """
    Convierte una imagen de OpenCV (numpy) en un PhotoImage de tkinter.
    Redimensiona la imagen para que entre en un cuadro de ancho_maximo x alto_maximo
    manteniendo la proporción original (ni panorámicas ni verticales se distorsionan).
    """

    if imagen_opencv is None:
        return None

    # OpenCV usa BGR, tkinter/PIL esperan RGB (solo si la imagen es a color)
    if len(imagen_opencv.shape) == 3:
        imagen_rgb = cv2.cvtColor(imagen_opencv, cv2.COLOR_BGR2RGB)
    else:
        imagen_rgb = imagen_opencv

    imagen_pil = Image.fromarray(imagen_rgb)

    # Calculamos un factor de escala para que la imagen entre en el cuadro
    ancho_original, alto_original = imagen_pil.size
    factor_ancho = ancho_maximo / ancho_original
    factor_alto = alto_maximo / alto_original
    factor_final = min(factor_ancho, factor_alto, 1.0)  # nunca agrandamos

    if factor_final < 1.0:
        nuevo_ancho = max(1, int(ancho_original * factor_final))
        nuevo_alto = max(1, int(alto_original * factor_final))
        imagen_pil = imagen_pil.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    return ImageTk.PhotoImage(imagen_pil)


# ============================================================
# Histogramas con matplotlib
# ============================================================

def crear_figura_histograma():
    """Crea una figura de matplotlib lista para mostrar un histograma pequeño."""

    figura = Figure(figsize=(2.6, 1.5), dpi=80, facecolor="white")
    eje = figura.add_subplot(111)
    eje.set_xlim(0, 255)
    eje.tick_params(labelsize=6)
    eje.grid(True, alpha=0.25, linewidth=0.5)
    figura.tight_layout(pad=0.4)
    return figura, eje


def dibujar_histograma(eje, frecuencias, color_barras="gray"):
    """Dibuja (o redibuja) el histograma sobre el eje indicado."""

    eje.clear()
    valores_x = list(range(256))
    eje.bar(valores_x, frecuencias, color=color_barras, width=1.0)
    eje.set_xlim(0, 255)
    eje.tick_params(labelsize=6)
    eje.grid(True, alpha=0.25, linewidth=0.5)


# ============================================================
# Marco con scroll vertical
# ============================================================

class MarcoScrollable(tk.Frame):
    """
    Frame con scroll vertical. Los widgets se agregan a `contenido_interno`.
    El contenido interno ocupa automáticamente todo el ancho del canvas,
    así no queda un espacio vacío a la derecha.
    """

    def __init__(self, contenedor_padre, color_fondo="#ecf0f1", *args, **kwargs):
        super().__init__(contenedor_padre, bg=color_fondo, *args, **kwargs)

        self.canvas = tk.Canvas(
            self, borderwidth=0, highlightthickness=0, bg=color_fondo
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.contenido_interno = tk.Frame(self.canvas, bg=color_fondo)

        # Cuando el contenido interno cambia de tamaño, recalculamos la
        # región de scroll del canvas
        self.contenido_interno.bind(
            "<Configure>",
            lambda evento: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        # Creamos la ventana del canvas con el frame dentro
        self.id_ventana_interna = self.canvas.create_window(
            (0, 0), window=self.contenido_interno, anchor="nw"
        )

        # Cuando el canvas cambie de tamaño (resize de la ventana),
        # forzamos a que el frame interno tenga el mismo ancho.
        # Esto evita el espacio vacío del lado derecho.
        self.canvas.bind("<Configure>", self._ajustar_ancho_al_canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Scroll con la rueda del mouse
        self.canvas.bind_all("<MouseWheel>", self._scroll_con_rueda)

    def _ajustar_ancho_al_canvas(self, evento):
        self.canvas.itemconfig(self.id_ventana_interna, width=evento.width)

    def _scroll_con_rueda(self, evento):
        # En Windows, delta = 120 por cada "tick" de rueda
        unidades_a_mover = int(-1 * (evento.delta / 120))
        self.canvas.yview_scroll(unidades_a_mover, "units")


# ============================================================
# RangeSlider personalizado (un solo slider con dos manejadores)
# ============================================================

class RangeSlider(tk.Frame):
    """
    Slider con dos manejadores: uno para el valor MÍNIMO y otro para el MÁXIMO.
    Ambos se dibujan sobre la misma pista y se arrastran con el mouse.

    - El valor mínimo nunca puede superar al valor máximo.
    - Cada vez que el usuario arrastra un manejador, se llama al callback con
      los nuevos valores (valor_min, valor_max).
    """

    def __init__(self, contenedor_padre,
                 desde=0, hasta=255,
                 valor_min_inicial=0, valor_max_inicial=255,
                 ancho=240, color_activo="#3498db",
                 callback=None, color_fondo="white", **kwargs):

        super().__init__(contenedor_padre, bg=color_fondo, **kwargs)

        # Rangos y valores
        self.desde = desde
        self.hasta = hasta
        self.valor_min = valor_min_inicial
        self.valor_max = valor_max_inicial
        self.color_activo = color_activo
        self.callback = callback

        # Tamaños
        self.ancho_canvas = ancho
        self.alto_canvas = 32
        self.margen_lateral = 14
        self.ancho_pista = ancho - 2 * self.margen_lateral
        self.radio_thumb = 9

        # Etiqueta con los valores actuales (arriba del slider)
        self.label_valores = tk.Label(
            self, text=self._texto_valores(),
            bg=color_fondo, fg="#2c3e50",
            font=("Arial", 8, "bold")
        )
        self.label_valores.pack()

        # Canvas donde se dibuja la pista y los manejadores
        self.canvas = tk.Canvas(
            self,
            width=self.ancho_canvas, height=self.alto_canvas,
            bg=color_fondo, highlightthickness=0
        )
        self.canvas.pack()

        # Estado interno: qué thumb se está arrastrando
        self._thumb_activo = None

        self._dibujar_slider()

        # Eventos del mouse sobre el canvas
        self.canvas.bind("<Button-1>", self._al_hacer_click)
        self.canvas.bind("<B1-Motion>", self._al_arrastrar)
        self.canvas.bind("<ButtonRelease-1>", self._al_soltar)

    # -------- conversiones entre valor y coordenada X --------

    def _valor_a_coordenada_x(self, valor):
        proporcion = (valor - self.desde) / (self.hasta - self.desde)
        return self.margen_lateral + proporcion * self.ancho_pista

    def _coordenada_x_a_valor(self, x):
        # Recortamos x al rango válido de la pista
        x_limite_izq = self.margen_lateral
        x_limite_der = self.margen_lateral + self.ancho_pista
        x_ajustada = max(x_limite_izq, min(x_limite_der, x))
        proporcion = (x_ajustada - x_limite_izq) / self.ancho_pista
        return int(round(self.desde + proporcion * (self.hasta - self.desde)))

    # -------- dibujo --------

    def _texto_valores(self):
        return f"Min: {self.valor_min}    Max: {self.valor_max}"

    def _dibujar_slider(self):
        """Redibuja la pista y los manejadores según los valores actuales."""
        self.canvas.delete("all")
        y_centro = self.alto_canvas // 2

        # Pista de fondo (línea gris)
        self.canvas.create_line(
            self.margen_lateral, y_centro,
            self.margen_lateral + self.ancho_pista, y_centro,
            fill="#d5dbdb", width=5, capstyle="round"
        )

        # Sección activa entre min y max (color del canal)
        x_min = self._valor_a_coordenada_x(self.valor_min)
        x_max = self._valor_a_coordenada_x(self.valor_max)
        self.canvas.create_line(
            x_min, y_centro, x_max, y_centro,
            fill=self.color_activo, width=5, capstyle="round"
        )

        # Manejador del mínimo
        self.canvas.create_oval(
            x_min - self.radio_thumb, y_centro - self.radio_thumb,
            x_min + self.radio_thumb, y_centro + self.radio_thumb,
            fill=self.color_activo, outline="white", width=2
        )
        # Manejador del máximo
        self.canvas.create_oval(
            x_max - self.radio_thumb, y_centro - self.radio_thumb,
            x_max + self.radio_thumb, y_centro + self.radio_thumb,
            fill=self.color_activo, outline="white", width=2
        )

        # Actualizamos el texto de los valores
        self.label_valores.config(text=self._texto_valores())

    # -------- manejo del mouse --------

    def _al_hacer_click(self, evento):
        """
        Al hacer click determinamos cuál de los dos thumbs queda mas cerca
        y lo fijamos como el que se está arrastrando.
        """
        x_min = self._valor_a_coordenada_x(self.valor_min)
        x_max = self._valor_a_coordenada_x(self.valor_max)

        distancia_al_min = abs(evento.x - x_min)
        distancia_al_max = abs(evento.x - x_max)

        if distancia_al_min <= distancia_al_max:
            self._thumb_activo = "min"
        else:
            self._thumb_activo = "max"

        # Ya aplicamos un movimiento inmediato (como un click-drag instantáneo)
        self._al_arrastrar(evento)

    def _al_arrastrar(self, evento):
        if self._thumb_activo is None:
            return

        nuevo_valor = self._coordenada_x_a_valor(evento.x)

        if self._thumb_activo == "min":
            # el mínimo siempre debe ser menor al máximo
            if nuevo_valor < self.valor_max:
                self.valor_min = nuevo_valor
            else:
                self.valor_min = self.valor_max - 1
        else:
            if nuevo_valor > self.valor_min:
                self.valor_max = nuevo_valor
            else:
                self.valor_max = self.valor_min + 1

        self._dibujar_slider()

        if self.callback is not None:
            self.callback(self.valor_min, self.valor_max)

    def _al_soltar(self, evento):
        self._thumb_activo = None

    # -------- API pública --------

    def obtener_valores(self):
        return self.valor_min, self.valor_max

    def resetear(self):
        """Vuelve el slider a sus valores por defecto (0 y 255)."""
        self.valor_min = self.desde
        self.valor_max = self.hasta
        self._dibujar_slider()
