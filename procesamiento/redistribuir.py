
import numpy as np

def normalizar_canal(canal, valor_minimo, valor_maximo):

    if canal is None:
        return None

    if valor_maximo <= valor_minimo:
        return canal.copy()

    canal_en_float = canal.astype(np.float32)

    canal_recortado = np.clip(canal_en_float, valor_minimo, valor_maximo)

    diferencia_rango = valor_maximo - valor_minimo
    canal_redistribuido = (canal_recortado - valor_minimo) / diferencia_rango * 255

    canal_normalizado = canal_redistribuido.astype(np.uint8)

    return canal_normalizado
