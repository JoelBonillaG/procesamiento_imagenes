# ============================================================
# Módulo: threshold.py
# ============================================================
# Aplica una "binarización" (thresholding) a la imagen comprimida.
#
# Un threshold convierte una imagen en escala de grises en una imagen
# con solo dos valores: 0 (negro) y 255 (blanco).
# Esto es útil como preprocesamiento para Machine Learning cuando
# solo nos interesa la forma de los objetos y no los detalles de color.
#
# Regla aplicada:
#   - si pixel >= media de la imagen -> se pone en 255 (blanco)
#   - si pixel <  media de la imagen -> se pone en 0   (negro)

import numpy as np


def aplicar_threshold(imagen_comprimida):
    """
    Binariza la imagen usando la media de sus propios píxeles como umbral.

    Parámetros:
        imagen_comprimida -> matriz 2D en escala de grises (uint8)

    Retorna:
        tupla (imagen_binaria, valor_media)
            - imagen_binaria: matriz 2D uint8 con solo valores 0 y 255
            - valor_media: media calculada (para mostrarla en la interfaz)
    """

    # Validación básica
    if imagen_comprimida is None:
        return None, 0

    # Paso 1: calcular la media de todos los píxeles de la imagen
    # Esta media será nuestro umbral (threshold)
    valor_media = np.mean(imagen_comprimida)

    # Paso 2: aplicar la regla pixel >= media -> 255, pixel < media -> 0
    # np.where funciona como un if-else aplicado a toda la matriz a la vez
    imagen_binaria = np.where(
        imagen_comprimida >= valor_media,
        255,
        0
    ).astype(np.uint8)

    return imagen_binaria, valor_media
