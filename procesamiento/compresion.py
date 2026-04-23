# ============================================================
# Módulo: compresion.py
# ============================================================
# Realiza dos tareas relacionadas con la compresión de la imagen:
#
#   1) Convertir la imagen a color en una imagen en escala de grises
#      usando la fórmula de luminancia:  0.299R + 0.587G + 0.114B
#
#   2) Aplicar una compresión por bloques: dividir la imagen en bloques
#      de NxN y reemplazar cada bloque por el promedio de sus píxeles.
#
# Se usan los coeficientes 0.299/0.587/0.114 porque corresponden a la
# sensibilidad del ojo humano a cada color (ITU-R BT.601).

import numpy as np


def convertir_a_grises(imagen_color):
    """
    Convierte una imagen a color (formato BGR de OpenCV) a escala de grises
    aplicando la fórmula de luminancia:

        gris = 0.299 * R + 0.587 * G + 0.114 * B

    Retorna una matriz 2D uint8 con valores entre 0 y 255.
    """

    if imagen_color is None:
        return None

    # OpenCV guarda las imágenes en BGR, así que:
    #   - canal 0 -> azul
    #   - canal 1 -> verde
    #   - canal 2 -> rojo
    # Convertimos cada canal a float para poder multiplicar por los coeficientes
    canal_azul = imagen_color[:, :, 0].astype(np.float32)
    canal_verde = imagen_color[:, :, 1].astype(np.float32)
    canal_rojo = imagen_color[:, :, 2].astype(np.float32)

    # Aplicamos la fórmula de luminancia
    imagen_grises_float = (
        0.299 * canal_rojo +
        0.587 * canal_verde +
        0.114 * canal_azul
    )

    # La convertimos de nuevo a uint8 porque así se manejan las imágenes normales
    imagen_grises = imagen_grises_float.astype(np.uint8)

    return imagen_grises


def comprimir_por_bloques(imagen_grises, tamano_bloque):
    """
    Divide la imagen en bloques cuadrados de tamano_bloque x tamano_bloque
    y reemplaza cada bloque por el PROMEDIO de los píxeles que contiene.

    Esto es una forma simple de compresión con pérdida: la imagen pierde
    detalle pero mantiene su forma general.

    Parámetros:
        imagen_grises  -> matriz 2D en escala de grises
        tamano_bloque  -> entero, tamaño del lado del bloque (ej: 2, 4, 8)

    Retorna:
        imagen_comprimida con el mismo tamaño que la original
    """

    if imagen_grises is None:
        return None

    # Validación del tamaño de bloque
    if tamano_bloque <= 0:
        return imagen_grises.copy()

    # Obtenemos las dimensiones de la imagen
    alto_imagen, ancho_imagen = imagen_grises.shape

    # Creamos una imagen vacía del mismo tamaño para ir guardando el resultado
    imagen_comprimida = np.zeros_like(imagen_grises)

    # Recorremos la imagen saltando de "tamano_bloque" en "tamano_bloque"
    # Tanto en filas como en columnas.
    for fila_inicial in range(0, alto_imagen, tamano_bloque):
        for columna_inicial in range(0, ancho_imagen, tamano_bloque):

            # Calculamos los límites del bloque.
            # Usamos min() para no pasarnos del borde de la imagen
            # (el último bloque puede ser mas pequeño que tamano_bloque)
            fila_final = min(fila_inicial + tamano_bloque, alto_imagen)
            columna_final = min(columna_inicial + tamano_bloque, ancho_imagen)

            # Extraemos los píxeles del bloque actual
            bloque_actual = imagen_grises[fila_inicial:fila_final,
                                          columna_inicial:columna_final]

            # Calculamos el promedio de los píxeles del bloque
            promedio_del_bloque = int(np.mean(bloque_actual))

            # Asignamos ese promedio a todos los píxeles del bloque en la imagen final
            imagen_comprimida[fila_inicial:fila_final,
                              columna_inicial:columna_final] = promedio_del_bloque

    return imagen_comprimida
