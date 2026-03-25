"""
generar_graficos.py — Genera todas las imagenes para el reporte

Uso:
    python generar_graficos.py

Salida en graficos/:
    distribucion_clases.png   distribucion del dataset por clase
    arquitectura_modelo.png   diagrama de capas del modelo
    flujo_sistema.png         diagrama de flujo del sistema completo
    pipeline_prediccion.png   pipeline de preprocesamiento e inferencia
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from collections import Counter

IMAGES_DIR  = "images"
OUTPUT_DIR  = "graficos"
CLASES_ES   = {
    "sedan":      "Sedan",
    "hatchback":  "Hatchback",
    "suv":        "SUV",
    "pickup":     "Pickup",
    "van":        "Van",
    "truck":      "Camion",
    "bus":        "Autobus",
    "motorcycle": "Motocicleta",
    "bicycle":    "Bicicleta",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG    = "#0f1117"
SURFACE    = "#1a1d27"
BORDER     = "#2e3144"
ACCENT     = "#4f8ef7"
TEXT       = "#e8eaf0"
MUTED      = "#6b7280"
OK         = "#22c55e"
WARN       = "#f59e0b"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "font.family":       "sans-serif",
    "font.size":         11,
})


# ──────────────────────────────────────────
# 1. DISTRIBUCION DEL DATASET
# ──────────────────────────────────────────
def grafico_distribucion():
    clases, conteos = [], []
    for clase in CLASES_ES:
        carpeta = os.path.join(IMAGES_DIR, clase)
        if os.path.isdir(carpeta):
            n = len([f for f in os.listdir(carpeta)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            clases.append(CLASES_ES[clase])
            conteos.append(n)

    fig, ax = plt.subplots(figsize=(10, 5))

    colores = [OK if c >= 70 else WARN if c >= 40 else "#ef4444" for c in conteos]
    bars = ax.barh(clases, conteos, color=colores, height=0.6, zorder=2)

    for bar, val in zip(bars, conteos):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=TEXT, fontsize=10)

    ax.axvline(40,  color=WARN,   linestyle="--", linewidth=1, alpha=0.7, label="Minimo recomendado (40)")
    ax.axvline(80,  color=OK,     linestyle="--", linewidth=1, alpha=0.7, label="Objetivo (80)")

    ax.set_xlabel("Cantidad de imagenes")
    ax.set_title("Distribucion del dataset por clase", fontsize=13, color=TEXT, pad=12)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="x", zorder=1)
    ax.set_xlim(0, max(conteos) + 15)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "distribucion_clases.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {ruta}")


# ──────────────────────────────────────────
# 2. ARQUITECTURA DEL MODELO
# ──────────────────────────────────────────
def grafico_arquitectura():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    capas = [
        (0.5,  "Entrada\n160x160x3",  SURFACE,  BORDER),
        (2.3,  "Augmentation\n(flip, rot,\nzoom, brillo)",   "#1a2744", "#2a4a8a"),
        (4.1,  "preprocess\ninput\n[-1, 1]",   "#1a2744", "#2a4a8a"),
        (5.9,  "MobileNetV2\n(ImageNet)\ncongelado",         "#1a3020", "#2a6040"),
        (7.7,  "Global Avg\nPooling2D",         "#2a1a30", "#6040a0"),
        (9.2,  "Dense 256\nReLU + L2",          "#2a1a30", "#6040a0"),
        (10.7, "Dropout\n0.5",                  "#2a1a30", "#6040a0"),
        (12.0, "Dense 9\nSoftmax",              "#1a3020", "#2a9060"),
    ]

    for x, texto, bg, borde in capas:
        rect = FancyBboxPatch((x - 0.55, 0.9), 1.1, 2.2,
                              boxstyle="round,pad=0.08",
                              facecolor=bg, edgecolor=borde, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 2.0, texto, ha="center", va="center",
                fontsize=7.5, color=TEXT, linespacing=1.4)

    for i in range(len(capas) - 1):
        x1 = capas[i][0] + 0.55
        x2 = capas[i + 1][0] - 0.55
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    ax.text(6.5, 0.4, "Transfer Learning: pesos preentrenados en ImageNet (1.2M imagenes)",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")

    ax.set_title("Arquitectura del modelo — MobileNetV2 + cabeza de clasificacion",
                 fontsize=12, color=TEXT, pad=10)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "arquitectura_modelo.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Guardado: {ruta}")


# ──────────────────────────────────────────
# 3. DIAGRAMA DE FLUJO DEL SISTEMA
# ──────────────────────────────────────────
def grafico_flujo_sistema():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def caja(x, y, w, h, texto, color_bg, color_borde, fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color_bg, edgecolor=color_borde, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, texto, ha="center", va="center",
                fontsize=fontsize, color=TEXT, linespacing=1.4)

    def flecha(x1, y1, x2, y2, etiqueta=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
        if etiqueta:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, etiqueta, fontsize=8, color=MUTED)

    # Entrenamiento (izquierda)
    ax.text(3, 6.6, "ENTRENAMIENTO", ha="center", fontsize=10,
            color=MUTED, fontweight="bold", style="italic")
    caja(3,   6.0, 3.2, 0.7, "images/<clase>/\n(80 imagenes/clase)", SURFACE, BORDER)
    caja(3,   4.9, 3.2, 0.7, "Carga + Oversample\n(min. 40/clase)", "#1a2744", "#2a4a8a")
    caja(3,   3.8, 3.2, 0.7, "Train/Val Split 80/20\n+ Class Weights", "#1a2744", "#2a4a8a")
    caja(3,   2.7, 3.2, 0.7, "Fase 1: entrenar cabeza\n(base congelada)", "#1a3020", "#2a6040")
    caja(3,   1.6, 3.2, 0.7, "Fase 2: fine-tuning\n(ultimas 50 capas)", "#1a3020", "#2a6040")
    caja(3,   0.6, 3.2, 0.7, "modelo_vehiculos.keras\n+ labels.json", "#2a1a30", "#6040a0")

    for y1, y2 in [(5.65, 5.25), (5.25, 4.15), (4.15, 3.05), (3.05, 1.95), (1.95, 0.95)]:
        flecha(3, y1, 3, y2)

    # Inferencia (derecha)
    ax.text(9, 6.6, "INFERENCIA", ha="center", fontsize=10,
            color=MUTED, fontweight="bold", style="italic")
    caja(9,   6.0, 3.2, 0.7, "Usuario sube imagen\n(JPG / PNG / WEBP)", SURFACE, BORDER)
    caja(9,   4.9, 3.2, 0.7, "FastAPI /predict\nvalidacion de formato", "#1a2744", "#2a4a8a")
    caja(9,   3.8, 3.2, 0.7, "Preprocesamiento\nresize 160x160 + [0,255]", "#1a2744", "#2a4a8a")
    caja(9,   2.7, 3.2, 0.7, "model.predict()\nMobileNetV2", "#1a3020", "#2a6040")
    caja(9,   1.6, 3.2, 0.7, "Top-3 predicciones\n+ nivel de confianza", "#1a3020", "#2a6040")
    caja(9,   0.6, 3.2, 0.7, "Resultado en interfaz\nweb (index.html)", "#2a1a30", "#6040a0")

    for y1, y2 in [(5.65, 5.25), (5.25, 4.15), (4.15, 3.05), (3.05, 1.95), (1.95, 0.95)]:
        flecha(9, y1, 9, y2)

    # Conexion entre flujos
    flecha(4.6, 0.6, 7.4, 0.6, "")
    ax.text(6, 0.75, "carga el modelo", ha="center", fontsize=8, color=MUTED)

    # Divisor central
    ax.plot([6, 6], [0.1, 6.8], color=BORDER, linewidth=1, linestyle="--")

    ax.set_title("Diagrama de flujo — Entrenamiento e Inferencia",
                 fontsize=12, color=TEXT, pad=10)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "flujo_sistema.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Guardado: {ruta}")


# ──────────────────────────────────────────
# 4. PIPELINE DE PREPROCESAMIENTO
# ──────────────────────────────────────────
def grafico_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    pasos = [
        (1.1,  "Imagen\noriginal\n(cualquier\ntamano)", SURFACE, BORDER),
        (3.1,  "convert('RGB')\neliminar canal\nalpha", "#1a2744", "#2a4a8a"),
        (5.1,  "resize\n160 x 160\n(LANCZOS)", "#1a2744", "#2a4a8a"),
        (7.1,  "np.array\nfloat32\n[0, 255]", "#1a2744", "#2a4a8a"),
        (9.1,  "expand_dims\n(1, 160, 160, 3)\nbatch dim", "#1a3020", "#2a6040"),
        (11.1, "preprocess\n_input (modelo)\n[-1, 1]", "#1a3020", "#2a6040"),
    ]

    for x, texto, bg, borde in pasos:
        rect = FancyBboxPatch((x - 0.9, 0.6), 1.8, 2.3,
                              boxstyle="round,pad=0.1",
                              facecolor=bg, edgecolor=borde, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 1.75, texto, ha="center", va="center",
                fontsize=8.5, color=TEXT, linespacing=1.4)

    for i in range(len(pasos) - 1):
        x1 = pasos[i][0] + 0.9
        x2 = pasos[i + 1][0] - 0.9
        ax.annotate("", xy=(x2, 1.75), xytext=(x1, 1.75),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    ax.set_title("Pipeline de preprocesamiento de imagen",
                 fontsize=12, color=TEXT, pad=8)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "pipeline_prediccion.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Guardado: {ruta}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("Generando graficos para el reporte...")
    grafico_distribucion()
    grafico_arquitectura()
    grafico_flujo_sistema()
    grafico_pipeline()

    # Copiar training_history.png si existe
    import shutil
    if os.path.exists("training_history.png"):
        shutil.copy("training_history.png", os.path.join(OUTPUT_DIR, "training_history.png"))
        print(f"Copiado: graficos/training_history.png")

    print(f"\nTodos los graficos guardados en {OUTPUT_DIR}/")
