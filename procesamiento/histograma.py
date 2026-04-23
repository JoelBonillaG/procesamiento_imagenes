# ============================================================
# Módulo: histograma.py
# ============================================================
# Calcula el histograma de un canal/imagen en escala de grises.
#
# Un histograma nos dice cuántos píxeles tienen cada valor de
# intensidad (de 0 a 255). Sirve para ver cómo están distribuidos
# los niveles de gris dentro del canal.

import numpy as np


def calcular_histograma(canal):
    """
    Calcula el histograma de un canal en escala de grises.

    Parámetros:
        canal -> matriz numpy 2D con valores entre 0 y 255 (uint8)

    Retorna:
        un arreglo de 256 posiciones.
        En la posición i se guarda la cantidad de píxeles con
        intensidad i en el canal dado.
    """

    # Validación básica: si no hay canal no hay nada que calcular
    if canal is None:
        return None

    # np.histogram nos devuelve dos cosas: las frecuencias y los bordes de los bins
    # Nosotros solo necesitamos las frecuencias, por eso descartamos lo otro con "_"
    # Usamos 256 bins porque los niveles de intensidad van de 0 a 255
    frecuencias_por_intensidad, _ = np.histogram(
        canal.ravel(),      # aplanamos el canal para que sea un arreglo 1D
        bins=256,
        range=(0, 256)
    )

    return frecuencias_por_intensidad
