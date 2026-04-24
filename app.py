# ============================================================
# Ecualizador de Imágenes — Preprocesamiento para Machine Learning
# ============================================================
# Punto de entrada de la aplicación.
# Configura customtkinter en modo oscuro y lanza la ventana principal.

import customtkinter as ctk
from interfaz.ventana import VentanaPrincipal


def main():
    # Configurar el tema ANTES de crear cualquier widget CTk
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    raiz = ctk.CTk()
    raiz.title("Ecualizador de Imágenes — Preprocesamiento ML")
    raiz.geometry("1420x920")
    raiz.minsize(1100, 700)

    VentanaPrincipal(raiz)

    raiz.mainloop()


if __name__ == "__main__":
    main()
