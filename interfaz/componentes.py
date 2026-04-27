
import tkinter as tk
import cv2
from PIL import Image, ImageTk

_TRACK_COLOR = "#cbd5e1"
_BG_COLOR    = "#ffffff"

class RangeSlider(tk.Frame):

    def __init__(self, contenedor_padre, desde=0, hasta=255,
                 valor_min_inicial=0, valor_max_inicial=255,
                 ancho=280, color_activo="#3b82f6",
                 callback=None, color_fondo=_BG_COLOR, **kwargs):

        super().__init__(contenedor_padre, bg=color_fondo, **kwargs)

        self.desde         = desde
        self.hasta         = hasta
        self.valor_min     = valor_min_inicial
        self.valor_max     = valor_max_inicial
        self.color_activo  = color_activo
        self.callback      = callback

        self.ancho_canvas  = ancho
        self.alto_canvas   = 26
        self.margen        = 10
        self.ancho_pista   = ancho - 2 * self.margen
        self.radio         = 7

        self.label_valores = tk.Label(
            self, text=self._texto_valores(),
            bg=color_fondo, fg="#475569",
            font=("Consolas", 8)
        )
        self.label_valores.pack()

        self.canvas = tk.Canvas(
            self,
            width=self.ancho_canvas, height=self.alto_canvas,
            bg=color_fondo, highlightthickness=0
        )
        self.canvas.pack()

        self._thumb_activo = None
        self._dibujar()

        self.canvas.bind("<Button-1>",        self._al_click)
        self.canvas.bind("<B1-Motion>",       self._al_arrastrar)
        self.canvas.bind("<ButtonRelease-1>", self._al_soltar)

    def _valor_a_x(self, valor):
        p = (valor - self.desde) / (self.hasta - self.desde)
        return self.margen + p * self.ancho_pista

    def _x_a_valor(self, x):
        x = max(self.margen, min(self.margen + self.ancho_pista, x))
        p = (x - self.margen) / self.ancho_pista
        return int(round(self.desde + p * (self.hasta - self.desde)))

    def _texto_valores(self):
        return f"Min {self.valor_min}   Max {self.valor_max}"

    def _dibujar(self):
        self.canvas.delete("all")
        y    = self.alto_canvas // 2
        x_mn = self._valor_a_x(self.valor_min)
        x_mx = self._valor_a_x(self.valor_max)
        r    = self.radio

        self.canvas.create_line(
            self.margen, y, self.margen + self.ancho_pista, y,
            fill=_TRACK_COLOR, width=4, capstyle="round"
        )
        self.canvas.create_line(
            x_mn, y, x_mx, y,
            fill=self.color_activo, width=4, capstyle="round"
        )
        self.canvas.create_oval(
            x_mn - r, y - r, x_mn + r, y + r,
            fill=self.color_activo, outline="#ffffff", width=2
        )
        self.canvas.create_oval(
            x_mx - r, y - r, x_mx + r, y + r,
            fill=self.color_activo, outline="#ffffff", width=2
        )
        self.label_valores.config(text=self._texto_valores())

    def _al_click(self, evento):
        dist_min = abs(evento.x - self._valor_a_x(self.valor_min))
        dist_max = abs(evento.x - self._valor_a_x(self.valor_max))
        self._thumb_activo = "min" if dist_min <= dist_max else "max"
        self._al_arrastrar(evento)

    def _al_arrastrar(self, evento):
        if not self._thumb_activo:
            return
        nuevo = self._x_a_valor(evento.x)
        if self._thumb_activo == "min":
            self.valor_min = min(nuevo, self.valor_max - 1)
        else:
            self.valor_max = max(nuevo, self.valor_min + 1)
        self._dibujar()
        if self.callback:
            self.callback(self.valor_min, self.valor_max)

    def _al_soltar(self, _):
        self._thumb_activo = None

    def obtener_valores(self):
        return self.valor_min, self.valor_max

    def set_valores(self, vmin, vmax):
        self.valor_min = vmin
        self.valor_max = vmax
        self._dibujar()

    def resetear(self):
        self.set_valores(self.desde, self.hasta)
        if self.callback:
            self.callback(self.valor_min, self.valor_max)

def convertir_imagen_para_tkinter(imagen_opencv, ancho_maximo=300, alto_maximo=240,
                                  expandir=False):
    if imagen_opencv is None:
        return None

    if len(imagen_opencv.shape) == 3:
        imagen_rgb = cv2.cvtColor(imagen_opencv, cv2.COLOR_BGR2RGB)
    else:
        imagen_rgb = imagen_opencv

    imagen_pil = Image.fromarray(imagen_rgb)

    ancho_original, alto_original = imagen_pil.size

    factor_escala = min(ancho_maximo / ancho_original, alto_maximo / alto_original)
    if not expandir:
        factor_escala = min(factor_escala, 1.0)

    if factor_escala != 1.0:
        nuevo_ancho = max(1, int(ancho_original * factor_escala))
        nuevo_alto  = max(1, int(alto_original  * factor_escala))
        resample = Image.NEAREST if factor_escala > 1.0 else Image.LANCZOS
        imagen_pil  = imagen_pil.resize((nuevo_ancho, nuevo_alto), resample)

    return ImageTk.PhotoImage(imagen_pil)
