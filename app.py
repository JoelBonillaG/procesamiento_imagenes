# ============================================================
# Ecualizador de Imágenes - Preprocesamiento para Machine Learning
# ============================================================
# Archivo principal de la aplicación.
# Se encarga de crear la ventana de tkinter y lanzar la interfaz.

import tkinter as tk
from interfaz.ventana import VentanaPrincipal


def main():
    # Creamos la ventana raíz de tkinter (la ventana del sistema operativo)
    ventana_raiz = tk.Tk()
    ventana_raiz.title("Ecualizador de Imágenes - Preprocesamiento ML")
    ventana_raiz.geometry("1500x920")
    ventana_raiz.minsize(1200, 700)

    # Instanciamos nuestra clase principal que construye toda la interfaz
    aplicacion = VentanaPrincipal(ventana_raiz)

    # Iniciamos el loop principal de tkinter (espera eventos del usuario)
    ventana_raiz.mainloop()


if __name__ == "__main__":
    main()
