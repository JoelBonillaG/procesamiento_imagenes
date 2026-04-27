
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_COLOR_FONDO_FIG = "#141720"
_COLOR_FONDO_EJE = "#141720"
_COLOR_BORDE_EJE = "#2d3152"
_COLOR_TICK      = "#5a6478"
_COLOR_LINEA     = "#e2e8f0"

class CurvaCanal:

    _RADIO_DETECCION = 20

    def __init__(self, contenedor_padre, color_canal="#ef4444",
                 callback=None, ancho_pulgadas=3.5, alto_pulgadas=2.6):
        self.color_canal  = color_canal
        self.callback     = callback
        self.valor_min    = 0
        self.valor_max    = 255
        self.frecuencias  = np.zeros(256)
        self._arrastrando = None

        self._figura = Figure(
            figsize=(ancho_pulgadas, alto_pulgadas),
            dpi=80,
            facecolor=_COLOR_FONDO_FIG
        )
        self._eje = self._figura.add_subplot(111)
        self._estilizar_ejes()
        self._figura.tight_layout(pad=0.5)

        self._canvas_mpl = FigureCanvasTkAgg(self._figura, master=contenedor_padre)
        self.widget = self._canvas_mpl.get_tk_widget()

        self._figura.canvas.mpl_connect("button_press_event",   self._al_presionar)
        self._figura.canvas.mpl_connect("motion_notify_event",  self._al_mover)
        self._figura.canvas.mpl_connect("button_release_event", self._al_soltar)

        self._dibujar()

    def pack(self, **kwargs):
        self.widget.pack(**kwargs)

    def grid(self, **kwargs):
        self.widget.grid(**kwargs)

    def cargar_histograma(self, frecuencias: np.ndarray):
        self.frecuencias = frecuencias.copy()
        self._dibujar()

    def set_valores(self, valor_min: int, valor_max: int):
        self.valor_min = valor_min
        self.valor_max = valor_max
        self._dibujar()

    def resetear(self):
        self.valor_min = 0
        self.valor_max = 255
        self._dibujar()
        if self.callback:
            self.callback(0, 255)

    def obtener_valores(self):
        return self.valor_min, self.valor_max

    def _estilizar_ejes(self):
        eje = self._eje
        eje.set_facecolor(_COLOR_FONDO_EJE)
        eje.tick_params(labelsize=6, colors=_COLOR_TICK)
        for borde in eje.spines.values():
            borde.set_color(_COLOR_BORDE_EJE)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)
        eje.set_xlim(0, 255)
        eje.set_ylim(0, 255)
        eje.set_xlabel("Entrada  (valor original del píxel)", fontsize=6, color=_COLOR_TICK)
        eje.set_ylabel("Salida  (valor normalizado)", fontsize=6, color=_COLOR_TICK)

    def _dibujar(self):
        eje = self._eje
        eje.clear()
        self._estilizar_ejes()

        if self.frecuencias.max() > 0:
            hist_escalado = (self.frecuencias / self.frecuencias.max()) * 230
        else:
            hist_escalado = self.frecuencias

        eje.bar(range(256), hist_escalado,
                color=self.color_canal, width=1.0, alpha=0.25)

        puntos_x = [0, self.valor_min, self.valor_max, 255]
        puntos_y = [0, 0,             255,             255]
        eje.plot(puntos_x, puntos_y, color=_COLOR_LINEA, linewidth=1.8, zorder=4)

        eje.plot(self.valor_min, 0, "o",
                 color=self.color_canal, markersize=10, zorder=6,
                 markeredgecolor="white", markeredgewidth=1.5)

        eje.plot(self.valor_max, 255, "o",
                 color=self.color_canal, markersize=10, zorder=6,
                 markeredgecolor="white", markeredgewidth=1.5)

        eje.grid(True, alpha=0.10, color="white", linewidth=0.5)

        self._canvas_mpl.draw_idle()

    def _punto_mas_cercano(self, x_dato, y_dato):
        distancia_al_min = np.hypot(x_dato - self.valor_min, y_dato - 0)
        distancia_al_max = np.hypot(x_dato - self.valor_max, y_dato - 255)

        if distancia_al_min <= distancia_al_max and distancia_al_min < self._RADIO_DETECCION:
            return "min"
        if distancia_al_max < distancia_al_min and distancia_al_max < self._RADIO_DETECCION:
            return "max"
        return None

    def _al_presionar(self, evento):
        if evento.inaxes != self._eje or evento.xdata is None:
            return
        self._arrastrando = self._punto_mas_cercano(evento.xdata, evento.ydata)

    def _al_mover(self, evento):
        if not self._arrastrando or evento.xdata is None:
            return

        x_nuevo = int(round(max(0.0, min(255.0, evento.xdata))))

        if self._arrastrando == "min":
            self.valor_min = min(x_nuevo, self.valor_max - 1)
        else:
            self.valor_max = max(x_nuevo, self.valor_min + 1)

        self._dibujar()

        if self.callback:
            self.callback(self.valor_min, self.valor_max)

    def _al_soltar(self, _evento):
        self._arrastrando = None
