# ============================================================
# Módulo: threshold.py
# ============================================================
# Binarización de imagen por umbral (threshold).
#
# Regla:
#   pixel >= umbral  → 255 (blanco)
#   pixel <  umbral  → 0   (negro)
#
# El umbral puede ser fijo (definido por el usuario con un slider)
# o automático (se usa la media de la imagen si no se pasa ninguno).
#
# Esta técnica simplifica la imagen para que un modelo de ML
# solo trabaje con la forma de los objetos (sin tonos de gris).

import numpy as np


def aplicar_threshold(imagen_comprimida, umbral=None):
    """
    Binariza la imagen usando el umbral indicado.

    Parámetros:
        imagen_comprimida -> matriz 2D en escala de grises (uint8)
        umbral            -> valor entre 0 y 255, o None para usar la media

    Retorna:
        tupla (imagen_binaria, media_calculada, umbral_usado)
            - imagen_binaria  : matriz 2D uint8 con solo valores 0 ó 255
            - media_calculada : media de la imagen (referencia informativa)
            - umbral_usado    : umbral que se aplicó efectivamente
    """

    if imagen_comprimida is None:
        return None, 0.0, 0

    # Siempre calculamos la media como referencia informativa para el usuario
    media_calculada = float(np.mean(imagen_comprimida))

    # Si no se especificó umbral, usamos la media de la imagen
    if umbral is None:
        umbral_usado = int(round(media_calculada))
    else:
        umbral_usado = int(umbral)

    # Aplicamos la binarización:
    # np.where recorre toda la matriz a la vez (más eficiente que un bucle)
    imagen_binaria = np.where(
        imagen_comprimida >= umbral_usado,
        255,   # blanco: el píxel está por encima del umbral
        0      # negro:  el píxel está por debajo del umbral
    ).astype(np.uint8)

    return imagen_binaria, media_calculada, umbral_usado
