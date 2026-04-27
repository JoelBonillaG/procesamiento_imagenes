
import numpy as np

def calcular_histograma(canal):

    if canal is None:
        return None

    frecuencias_por_intensidad, _ = np.histogram(
        canal.ravel(),
        bins=256,
        range=(0, 256)
    )

    return frecuencias_por_intensidad
