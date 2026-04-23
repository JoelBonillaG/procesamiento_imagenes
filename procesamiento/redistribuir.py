# ============================================================
# Módulo: redistribuir.py
# ============================================================
# Realiza la NORMALIZACIÓN MIN-MAX de un canal.
#
# La idea es que el usuario elige un valor mínimo y uno máximo,
# y los píxeles del canal se redistribuyen para que esos valores
# ocupen todo el rango 0-255.
#
# Esto sirve como técnica de preprocesamiento para Machine Learning,
# porque nos permite mejorar el contraste del canal de una imagen.
#
# IMPORTANTE: el histograma NO se desplaza, se REDISTRIBUYE
# (se "estira" el rango seleccionado hasta ocupar todo el 0-255).

import numpy as np


def normalizar_canal(canal, valor_minimo, valor_maximo):
    """
    Aplica normalización min-max a un canal (matriz 2D de valores 0-255).

    Reglas:
        - Si el pixel es menor que valor_minimo  -> se vuelve 0
        - Si el pixel es mayor que valor_maximo  -> se vuelve 255
        - Si está entre ambos -> se redistribuye con la fórmula:
              pixel_nuevo = (pixel - min) / (max - min) * 255

    Parámetros:
        canal         -> matriz numpy 2D con valores entre 0 y 255
        valor_minimo  -> umbral inferior elegido por el usuario
        valor_maximo  -> umbral superior elegido por el usuario

    Retorna:
        nueva matriz 2D con los valores normalizados
    """

    # Validación: que el canal exista
    if canal is None:
        return None

    # Validación: si el usuario puso mal los valores, evitamos la división por cero
    # y simplemente devolvemos una copia del canal sin modificar
    if valor_maximo <= valor_minimo:
        return canal.copy()

    # Convertimos a float32 para poder hacer operaciones matemáticas sin
    # que se desborden los valores (uint8 solo admite 0-255)
    canal_en_float = canal.astype(np.float32)

    # Paso 1: recortamos los valores que estén fuera del rango
    # Todo lo que esté por debajo del mínimo queda en el mínimo
    # Todo lo que esté por encima del máximo queda en el máximo
    canal_recortado = np.clip(canal_en_float, valor_minimo, valor_maximo)

    # Paso 2: aplicamos la fórmula de normalización min-max
    # (pixel - min) / (max - min) * 255
    diferencia_rango = valor_maximo - valor_minimo
    canal_redistribuido = (canal_recortado - valor_minimo) / diferencia_rango * 255

    # Paso 3: regresamos el canal a uint8 (tipo estándar de imágenes)
    canal_normalizado = canal_redistribuido.astype(np.uint8)

    return canal_normalizado
