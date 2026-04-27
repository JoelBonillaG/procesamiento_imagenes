
import numpy as np

def aplicar_threshold(imagen_comprimida, umbral=None):

    if imagen_comprimida is None:
        return None, 0.0, 0

    media_calculada = float(np.mean(imagen_comprimida))

    if umbral is None:
        umbral_usado = int(round(media_calculada))
    else:
        umbral_usado = int(umbral)

    imagen_binaria = np.where(
        imagen_comprimida >= umbral_usado,
        255,
        0
    ).astype(np.uint8)

    return imagen_binaria, media_calculada, umbral_usado
