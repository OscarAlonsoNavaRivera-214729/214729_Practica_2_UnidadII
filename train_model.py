"""
train_model.py — Entrenador del clasificador de vehiculos
Arquitectura: Transfer Learning con MobileNetV2 + fine-tuning en dos fases

Uso:
    python train_model.py

Estructura de imagenes:
    images/<clase>/   (min. 40 imagenes por clase recomendado)
"""

import os
import json
import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────
# CONFIGURACION
# ──────────────────────────────────────────
IMG_SIZE         = (160, 160)
BATCH_SIZE       = 16
EPOCHS_1         = 20
EPOCHS_2         = 20
LR_1             = 1e-3
LR_2             = 2e-5
TARGET_POR_CLASE = 40       # oversample si una clase tiene menos
IMAGES_DIR       = "images"
MODEL_PATH       = "modelo_vehiculos.keras"
LABELS_PATH      = "labels.json"

CLASES = [
    "sedan", "hatchback", "suv", "pickup",
    "van", "truck", "bus", "motorcycle", "bicycle"
]


# ──────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────
def leer_imagen(ruta):
    img = Image.open(ruta).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype="float32")  # [0, 255]


def oversample(imagenes, objetivo):
    """Completa una lista hasta 'objetivo' con flip + variacion de brillo."""
    resultado = list(imagenes)
    while len(resultado) < objetivo:
        src = resultado[np.random.randint(len(resultado))].copy()
        if np.random.random() > 0.5:
            src = src[:, ::-1, :]
        src = np.clip(src * np.random.uniform(0.85, 1.15), 0, 255)
        resultado.append(src)
    return resultado


# ──────────────────────────────────────────
# PASO 1 — CARGAR DATASET
# Imagenes en [0, 255]. preprocess_input dentro del modelo aplica [-1, 1].
# Clases con menos de TARGET_POR_CLASE muestras se completan con oversample.
# ──────────────────────────────────────────
def cargar_dataset(clases_disponibles):
    print("\nCargando imagenes...")
    X, y = [], []

    for idx, clase in enumerate(clases_disponibles):
        carpeta = os.path.join(IMAGES_DIR, clase)
        archivos = [f for f in os.listdir(carpeta)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        imagenes = []
        for archivo in archivos:
            try:
                imagenes.append(leer_imagen(os.path.join(carpeta, archivo)))
            except Exception:
                pass

        originales = len(imagenes)
        if len(imagenes) < TARGET_POR_CLASE:
            imagenes = oversample(imagenes, TARGET_POR_CLASE)

        extra = len(imagenes) - originales
        sufijo = f" (+ {extra} oversample)" if extra else ""
        print(f"  {clase:<12} {len(imagenes):>4} imagenes{sufijo}")

        X.extend(imagenes)
        y.extend([idx] * len(imagenes))

    X = np.array(X, dtype="float32")  # (N, 160, 160, 3)
    y = np.array(y)
    print(f"\nTotal muestras: {len(X)}")
    return X, y


# ──────────────────────────────────────────
# PASO 2 — MODELO: MobileNetV2 + cabeza
# ──────────────────────────────────────────
def construir_modelo(num_clases, input_shape):
    print(f"\nConstruyendo modelo (MobileNetV2) — {num_clases} clases | entrada {input_shape}")

    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.RandomTranslation(0.10, 0.10)(x)
    x = layers.RandomContrast(0.10)(x)
    x = layers.RandomBrightness(0.10)(x)

    # [0, 255] -> [-1, 1] requerido por MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_clases, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="vehiculo_mobilenetv2"), base


# ──────────────────────────────────────────
# PASO 3 — COMPILAR
# ──────────────────────────────────────────
def compilar_modelo(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


# ──────────────────────────────────────────
# PASO 4 — ENTRENAMIENTO EN DOS FASES
# Fase 1: cabeza con base congelada, LR alto para converger rapido
# Fase 2: fine-tuning de ultimas 50 capas con LR bajo
# ──────────────────────────────────────────
def entrenar(model, base, X_train, y_train, X_val, y_val, class_weight):

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        ),
    ]

    print(f"\nFASE 1 — Cabeza con base congelada ({EPOCHS_1} epocas max, LR={LR_1})")
    compilar_modelo(model, LR_1)
    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_1,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nFASE 2 — Fine-tuning ultimas 50 capas ({EPOCHS_2} epocas max, LR={LR_2})")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    compilar_modelo(model, LR_2)
    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_2,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    combined = {}
    for k in history1.history:
        combined[k] = history1.history[k] + history2.history[k]
    return combined


# ──────────────────────────────────────────
# PASO 5 — EVALUACION
# ──────────────────────────────────────────
def evaluar(model, X_val, y_val, clases_disponibles):
    print("\nReporte de clasificacion:")
    preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(classification_report(y_val, preds, target_names=clases_disponibles))


# ──────────────────────────────────────────
# PASO 6 — GUARDAR ETIQUETAS
# ──────────────────────────────────────────
def guardar_labels(clases_disponibles):
    labels = {str(i): clase for i, clase in enumerate(clases_disponibles)}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Etiquetas guardadas en {LABELS_PATH}")


# ──────────────────────────────────────────
# PASO 7 — GRAFICA DE ENTRENAMIENTO
# ──────────────────────────────────────────
def graficar(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["accuracy"],     label="Entrenamiento")
    axes[0].plot(history["val_accuracy"], label="Validacion")
    axes[0].set_title("Exactitud por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["loss"],     label="Entrenamiento")
    axes[1].plot(history["val_loss"], label="Validacion")
    axes[1].set_title("Perdida por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Grafica guardada en training_history.png")
    plt.show()


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Clasificador de Vehiculos - MobileNetV2")
    print("=" * 55)

    clases_disponibles = []
    for clase in CLASES:
        ruta = os.path.join(IMAGES_DIR, clase)
        if os.path.isdir(ruta):
            imgs = [f for f in os.listdir(ruta)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if imgs:
                clases_disponibles.append(clase)

    if len(clases_disponibles) < 2:
        print("Necesitas al menos 2 clases con imagenes en images/<clase>/")
        exit(1)

    print(f"Clases: {clases_disponibles}")

    X, y = cargar_dataset(clases_disponibles)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))
    print(f"Class weights: { {clases_disponibles[i]: round(v, 2) for i, v in class_weight.items()} }")

    model, base = construir_modelo(len(clases_disponibles), (*IMG_SIZE, 3))
    history = entrenar(model, base, X_train, y_train, X_val, y_val, class_weight)
    evaluar(model, X_val, y_val, clases_disponibles)
    guardar_labels(clases_disponibles)
    graficar(history)

    print("\n" + "=" * 55)
    print("  Listo. Ejecuta: fastapi dev main.py")
    print("=" * 55)
