
import customtkinter as ctk
from interfaz.ventana import VentanaPrincipal

def main():
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
