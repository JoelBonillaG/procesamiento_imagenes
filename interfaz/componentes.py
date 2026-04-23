# ============================================================
# Módulo: componentes.py
# ============================================================
# Función utilitaria de conversión de imágenes OpenCV → PhotoImage de tkinter.
# La interfaz ahora usa customtkinter para todo el layout y el estilo,
# por lo que este módulo se reduce a la conversión de imágenes.

import cv2
from PIL import Image, ImageTk


def convertir_imagen_para_tkinter(imagen_opencv, ancho_maximo=300, alto_maximo=240):
    """
    Convierte una imagen de OpenCV (numpy BGR o escala de grises)
    a un PhotoImage que tkinter puede mostrar en un Label.

    Redimensiona proporcionalmente para que la imagen quepa dentro del
    cuadro de ancho_maximo × alto_maximo píxeles, sin agrandarla nunca.

    Parámetros:
        imagen_opencv -> matriz numpy 2D (grises) o 3D (BGR)
        ancho_maximo  -> ancho máximo del cuadro destino en píxeles
        alto_maximo   -> alto máximo del cuadro destino en píxeles

    Retorna:
        ImageTk.PhotoImage listo para asignar a un Label con label.config(image=...)
    """
    if imagen_opencv is None:
        return None

    # OpenCV usa BGR; PIL espera RGB — hacemos la conversión si hay 3 canales
    if len(imagen_opencv.shape) == 3:
        imagen_rgb = cv2.cvtColor(imagen_opencv, cv2.COLOR_BGR2RGB)
    else:
        imagen_rgb = imagen_opencv   # escala de grises: no necesita conversión

    imagen_pil = Image.fromarray(imagen_rgb)

    ancho_original, alto_original = imagen_pil.size

    # Calculamos el factor de escala que hace que la imagen quepa en ambas dimensiones
    factor_escala = min(
        ancho_maximo / ancho_original,
        alto_maximo  / alto_original,
        1.0   # nunca agrandamos la imagen, solo la encogemos
    )

    if factor_escala < 1.0:
        nuevo_ancho = max(1, int(ancho_original * factor_escala))
        nuevo_alto  = max(1, int(alto_original  * factor_escala))
        imagen_pil  = imagen_pil.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    return ImageTk.PhotoImage(imagen_pil)
