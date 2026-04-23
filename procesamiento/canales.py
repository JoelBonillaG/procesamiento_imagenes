# ============================================================
# Módulo: canales.py
# ============================================================
# Se encarga de separar una imagen a color en sus 3 canales (R, G, B)
# y también de unirlos de nuevo en una sola imagen.
#
# Importante: OpenCV carga las imágenes en formato BGR (no RGB),
# por eso hay que tener cuidado con el orden de los canales al
# separarlos y al volverlos a unir.

import cv2


def separar_canales(imagen_original):
    """
    Recibe una imagen a color (cargada con OpenCV, formato BGR)
    y devuelve los 3 canales por separado en orden R, G, B.

    Cada canal es una matriz 2D en escala de grises donde cada valor
    representa la intensidad de ese color en el píxel correspondiente.
    """

    # Validamos que realmente haya una imagen
    if imagen_original is None:
        return None, None, None

    # cv2.split nos devuelve los canales en el orden en que los guarda OpenCV: BGR
    canal_azul, canal_verde, canal_rojo = cv2.split(imagen_original)

    # Los devolvemos en el orden lógico (R, G, B) para que sea mas claro al usar
    return canal_rojo, canal_verde, canal_azul


def unir_canales(canal_rojo, canal_verde, canal_azul):
    """
    Recibe los 3 canales R, G, B por separado y los combina en una sola
    imagen a color. El resultado estará en formato BGR (porque así lo usa OpenCV).
    """

    # Validaciones básicas para evitar que se rompa el programa
    if canal_rojo is None or canal_verde is None or canal_azul is None:
        return None

    # cv2.merge espera los canales en formato BGR (azul primero, rojo al final)
    imagen_recombinada = cv2.merge([canal_azul, canal_verde, canal_rojo])

    return imagen_recombinada
