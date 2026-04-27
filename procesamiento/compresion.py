
import numpy as np

def convertir_a_grises(imagen_color):

    if imagen_color is None:
        return None

    canal_azul = imagen_color[:, :, 0].astype(np.float32)
    canal_verde = imagen_color[:, :, 1].astype(np.float32)
    canal_rojo = imagen_color[:, :, 2].astype(np.float32)

    imagen_grises_float = (
        0.299 * canal_rojo +
        0.587 * canal_verde +
        0.114 * canal_azul
    )

    imagen_grises = imagen_grises_float.astype(np.uint8)

    return imagen_grises

def comprimir_por_bloques(imagen_grises, tamano_bloque):

    if imagen_grises is None:
        return None

    if tamano_bloque <= 0:
        return imagen_grises.copy()

    alto_imagen, ancho_imagen = imagen_grises.shape

    alto_salida = (alto_imagen + tamano_bloque - 1) // tamano_bloque
    ancho_salida = (ancho_imagen + tamano_bloque - 1) // tamano_bloque
    imagen_comprimida = np.zeros((alto_salida, ancho_salida), dtype=np.uint8)

    fila_salida = 0
    for fila_inicial in range(0, alto_imagen, tamano_bloque):
        columna_salida = 0
        for columna_inicial in range(0, ancho_imagen, tamano_bloque):

            fila_final = min(fila_inicial + tamano_bloque, alto_imagen)
            columna_final = min(columna_inicial + tamano_bloque, ancho_imagen)

            bloque_actual = imagen_grises[fila_inicial:fila_final,
                                          columna_inicial:columna_final]

            promedio_del_bloque = int(np.mean(bloque_actual))

            imagen_comprimida[fila_salida, columna_salida] = promedio_del_bloque
            columna_salida += 1

        fila_salida += 1

    return imagen_comprimida
