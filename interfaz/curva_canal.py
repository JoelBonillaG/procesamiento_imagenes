# ============================================================
# Módulo: curva_canal.py
# ============================================================
# Widget de curva interactiva estilo "Levels" de Photoshop.
#
# Permite al usuario ajustar visualmente la normalización
# de un canal de color arrastrando dos puntos con el mouse:
#
#   Punto MIN (círculo inferior izquierdo):
#       Define desde qué valor de entrada empieza el estiramiento.
#       Todos los píxeles con valor < min se vuelven 0 (negro total).
#
#   Punto MAX (círculo superior derecho):
#       Define hasta qué valor de entrada llega el estiramiento.
#       Todos los píxeles con valor > max se vuelven 255 (blanco total).
#
#   Para píxeles entre MIN y MAX se aplica:
#       pixel_nuevo = (pixel - min) / (max - min) × 255
#
# El histograma del canal original aparece de fondo (semitransparente)
# para que el usuario vea dónde está la distribución de la imagen.
#
# Cuando el usuario arrastra un punto se invoca callback(valor_min, valor_max).

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Paleta oscura interna de la curva ─────────────────────────────────────
_COLOR_FONDO_FIG = "#141720"   # fondo de la figura matplotlib
_COLOR_FONDO_EJE = "#141720"   # fondo del área de los ejes
_COLOR_BORDE_EJE = "#2d3152"   # color de los bordes (spines)
_COLOR_TICK      = "#5a6478"   # color de los números en los ejes
_COLOR_LINEA     = "#e2e8f0"   # color de la línea de transferencia


class CurvaCanal:
    """
    Curva interactiva de normalización por canal (tipo Levels de Photoshop).

    Se embebe en cualquier contenedor tkinter o customtkinter.
    Los métodos pack() y grid() delegan directamente al widget interno.
    """

    # Tolerancia en unidades del gráfico (escala 0-255) para detectar
    # si el clic del mouse cayó cerca de uno de los dos puntos
    _RADIO_DETECCION = 20

    def __init__(self, contenedor_padre, color_canal="#ef4444",
                 callback=None, ancho_pulgadas=3.5, alto_pulgadas=2.6):
        """
        Parámetros:
            contenedor_padre -> frame tkinter/CTk donde se inserta el widget
            color_canal      -> color hex del canal (rojo, verde o azul)
            callback         -> función(valor_min, valor_max) que se llama al cambiar
            ancho_pulgadas   -> ancho de la figura en pulgadas (dpi=80)
            alto_pulgadas    -> alto de la figura en pulgadas (dpi=80)
        """
        self.color_canal  = color_canal
        self.callback     = callback
        self.valor_min    = 0      # valor de entrada donde empieza el estiramiento
        self.valor_max    = 255    # valor de entrada donde termina el estiramiento
        self.frecuencias  = np.zeros(256)  # histograma del canal original (fondo)
        self._arrastrando = None   # 'min' o 'max' mientras el mouse está presionado

        # Crear figura matplotlib con estilo oscuro
        self._figura = Figure(
            figsize=(ancho_pulgadas, alto_pulgadas),
            dpi=80,
            facecolor=_COLOR_FONDO_FIG
        )
        self._eje = self._figura.add_subplot(111)
        self._estilizar_ejes()
        self._figura.tight_layout(pad=0.5)

        # Incrustar la figura en el contenedor tkinter
        self._canvas_mpl = FigureCanvasTkAgg(self._figura, master=contenedor_padre)
        self.widget = self._canvas_mpl.get_tk_widget()

        # Conectar los eventos del mouse al canvas de matplotlib
        self._figura.canvas.mpl_connect("button_press_event",   self._al_presionar)
        self._figura.canvas.mpl_connect("motion_notify_event",  self._al_mover)
        self._figura.canvas.mpl_connect("button_release_event", self._al_soltar)

        self._dibujar()

    # ── Métodos de geometría tkinter ─────────────────────────────────────

    def pack(self, **kwargs):
        self.widget.pack(**kwargs)

    def grid(self, **kwargs):
        self.widget.grid(**kwargs)

    # ── API pública ───────────────────────────────────────────────────────

    def cargar_histograma(self, frecuencias: np.ndarray):
        """
        Actualiza el histograma de fondo con los datos del canal original.
        No dispara el callback (solo actualiza la visualización).
        """
        self.frecuencias = frecuencias.copy()
        self._dibujar()

    def set_valores(self, valor_min: int, valor_max: int):
        """
        Establece los valores min y max sin disparar el callback.
        Útil para resetear la curva desde el exterior.
        """
        self.valor_min = valor_min
        self.valor_max = valor_max
        self._dibujar()

    def resetear(self):
        """Restablece min=0 y max=255 y sí dispara el callback."""
        self.valor_min = 0
        self.valor_max = 255
        self._dibujar()
        if self.callback:
            self.callback(0, 255)

    def obtener_valores(self):
        """Devuelve (valor_min, valor_max) actuales."""
        return self.valor_min, self.valor_max

    # ── Dibujo ────────────────────────────────────────────────────────────

    def _estilizar_ejes(self):
        """Aplica el estilo oscuro a los ejes de matplotlib."""
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
        """Redibuja la curva completa: histograma + línea de transferencia + puntos."""
        eje = self._eje
        eje.clear()
        self._estilizar_ejes()

        # Histograma de fondo: lo escalamos para que el pico llegue a ~230
        # (en las mismas unidades del eje Y, que va de 0 a 255)
        if self.frecuencias.max() > 0:
            hist_escalado = (self.frecuencias / self.frecuencias.max()) * 230
        else:
            hist_escalado = self.frecuencias

        eje.bar(range(256), hist_escalado,
                color=self.color_canal, width=1.0, alpha=0.25)

        # Línea de transferencia lineal por partes:
        #   (0, 0) → (min, 0) → (max, 255) → (255, 255)
        # Esta forma define exactamente la fórmula de normalización min-max
        puntos_x = [0, self.valor_min, self.valor_max, 255]
        puntos_y = [0, 0,             255,             255]
        eje.plot(puntos_x, puntos_y, color=_COLOR_LINEA, linewidth=1.8, zorder=4)

        # Punto MIN en la posición (valor_min, 0) — parte baja de la línea
        eje.plot(self.valor_min, 0, "o",
                 color=self.color_canal, markersize=10, zorder=6,
                 markeredgecolor="white", markeredgewidth=1.5)

        # Punto MAX en la posición (valor_max, 255) — parte alta de la línea
        eje.plot(self.valor_max, 255, "o",
                 color=self.color_canal, markersize=10, zorder=6,
                 markeredgecolor="white", markeredgewidth=1.5)

        # Cuadrícula sutil para facilitar la lectura de valores
        eje.grid(True, alpha=0.10, color="white", linewidth=0.5)

        self._canvas_mpl.draw_idle()

    # ── Eventos del mouse ─────────────────────────────────────────────────

    def _punto_mas_cercano(self, x_dato, y_dato):
        """
        Determina si el clic cayó dentro del radio de tolerancia
        de alguno de los dos puntos (min o max).
        """
        distancia_al_min = np.hypot(x_dato - self.valor_min, y_dato - 0)
        distancia_al_max = np.hypot(x_dato - self.valor_max, y_dato - 255)

        if distancia_al_min <= distancia_al_max and distancia_al_min < self._RADIO_DETECCION:
            return "min"
        if distancia_al_max < distancia_al_min and distancia_al_max < self._RADIO_DETECCION:
            return "max"
        return None

    def _al_presionar(self, evento):
        """El usuario presionó el botón del mouse: determina qué punto arrastrar."""
        if evento.inaxes != self._eje or evento.xdata is None:
            return
        self._arrastrando = self._punto_mas_cercano(evento.xdata, evento.ydata)

    def _al_mover(self, evento):
        """El usuario está arrastrando el mouse: actualiza el punto activo."""
        if not self._arrastrando or evento.xdata is None:
            return

        # Convertimos la posición X del mouse a un valor entero entre 0 y 255
        x_nuevo = int(round(max(0.0, min(255.0, evento.xdata))))

        if self._arrastrando == "min":
            # El punto mínimo no puede sobrepasar al máximo (dejamos al menos 1 de margen)
            self.valor_min = min(x_nuevo, self.valor_max - 1)
        else:
            # El punto máximo no puede quedar por debajo del mínimo
            self.valor_max = max(x_nuevo, self.valor_min + 1)

        self._dibujar()

        # Avisamos al observador (ventana principal) del nuevo rango
        if self.callback:
            self.callback(self.valor_min, self.valor_max)

    def _al_soltar(self, _evento):
        """El usuario soltó el botón del mouse: terminamos el arrastre."""
        self._arrastrando = None
