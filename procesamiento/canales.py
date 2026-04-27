
import cv2

def separar_canales(imagen_original):

    if imagen_original is None:
        return None, None, None

    canal_azul, canal_verde, canal_rojo = cv2.split(imagen_original)

    return canal_rojo, canal_verde, canal_azul

def unir_canales(canal_rojo, canal_verde, canal_azul):

    if canal_rojo is None or canal_verde is None or canal_azul is None:
        return None

    imagen_recombinada = cv2.merge([canal_azul, canal_verde, canal_rojo])

    return imagen_recombinada
