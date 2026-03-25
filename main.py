"""
main.py — API del Clasificador de Vehiculos
Uso:
    source venv/bin/activate
    fastapi dev main.py   ->  http://localhost:8000

Endpoints:
  GET  /        -> index.html
  POST /predict -> { prediccion_principal, top_3 }
  GET  /health  -> estado del servidor
  GET  /clases  -> clases disponibles
"""

import os
import json
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# ──────────────────────────────────────────
# CONFIGURACION — debe coincidir con train_model.py
# ──────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = str(BASE_DIR / "modelo_vehiculos.keras")
LABELS_PATH = str(BASE_DIR / "labels.json")
IMG_SIZE    = (160, 160)

DESCRIPCIONES = {
    "sedan": {
        "titulo": "Sedan",
        "descripcion": "Automovil de 4 puertas con 3 compartimentos separados. "
                       "El mas comun en ciudad. Ejemplos: Nissan Versa, Toyota Corolla, VW Jetta, Honda Civic."
    },
    "hatchback": {
        "titulo": "Hatchback",
        "descripcion": "Compacto con maletero integrado y puerta trasera elevable. "
                       "Practico en ciudad. Ejemplos: VW Golf, Toyota Yaris, Chevrolet Spark, Ford Fiesta."
    },
    "suv": {
        "titulo": "SUV / Crossover",
        "descripcion": "Vehiculo alto con mayor capacidad y traccion mejorada. "
                       "Popular para familias. Ejemplos: Toyota RAV4, Nissan Kicks, Honda CR-V, Mazda CX-5."
    },
    "pickup": {
        "titulo": "Pickup / Camioneta",
        "descripcion": "Cabina para pasajeros y caja descubierta para carga. "
                       "Muy popular en Mexico. Ejemplos: Ford F-150, RAM 1500, Chevrolet Silverado, Toyota Tacoma."
    },
    "van": {
        "titulo": "Van / Furgoneta",
        "descripcion": "Vehiculo cerrado de carga o pasajeros. Uso comercial o familiar. "
                       "Ejemplos: Mercedes Sprinter, Ford Transit, VW Transporter, RAM ProMaster."
    },
    "truck": {
        "titulo": "Camion",
        "descripcion": "Vehiculo de carga mediano o pesado para distribucion. "
                       "Ejemplos: camiones de 3.5 a 10 toneladas, Hino, Isuzu, International."
    },
    "bus": {
        "titulo": "Autobus",
        "descripcion": "Transporte publico o turistico de multiples pasajeros. "
                       "Ejemplos: autobuses urbanos, de linea, microbuses, minibuses."
    },
    "motorcycle": {
        "titulo": "Motocicleta",
        "descripcion": "Vehiculo de dos ruedas con motor. Agil y economico. "
                       "Ejemplos: Honda CB, Kawasaki Ninja, Harley-Davidson, Yamaha MT, scooters."
    },
    "bicycle": {
        "titulo": "Bicicleta",
        "descripcion": "Vehiculo de dos ruedas a pedal, sin motor. "
                       "Tipos: urbana, mountain bike, ruta, BMX, electrica."
    },
}

# ──────────────────────────────────────────
# ESTADO GLOBAL DEL MODELO
# ──────────────────────────────────────────
estado = {"model": None, "labels": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando servidor...")

    if os.path.exists(MODEL_PATH):
        try:
            import keras
            print(f"Cargando modelo: {MODEL_PATH}")
            estado["model"] = keras.saving.load_model(MODEL_PATH)
            print("Modelo listo")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
    else:
        print(f"Modelo no encontrado. Ejecuta: python train_model.py")

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            estado["labels"] = json.load(f)
        print(f"{len(estado['labels'])} clases cargadas")

    yield

    print("Cerrando...")


# ──────────────────────────────────────────
# APP
# ──────────────────────────────────────────
app = FastAPI(
    title="VehicleAI — Clasificador de Vehiculos",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# PREPROCESAMIENTO
# Imagenes en [0, 255] — preprocess_input dentro del modelo hace [-1, 1]
# ──────────────────────────────────────────
def preprocesar(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype="float32")   # [0, 255] — el modelo aplica preprocess_input
    return np.expand_dims(arr, axis=0)     # (1, 128, 128, 3)


# ──────────────────────────────────────────
# PREDICCION
# ──────────────────────────────────────────
def predecir(imagen_bytes: bytes) -> dict:
    model  = estado["model"]
    labels = estado["labels"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta python train_model.py primero."
        )

    arr = preprocesar(imagen_bytes)
    probs = model.predict(arr, verbose=0)[0]

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = []
    for idx in top_idx:
        clave = labels[str(idx)]
        info  = DESCRIPCIONES.get(clave, {
            "titulo": clave.replace("_", " ").title(),
            "descripcion": f"Vehiculo tipo {clave}"
        })
        top3.append({
            "clase":       clave,
            "titulo":      info["titulo"],
            "porcentaje":  round(float(probs[idx]) * 100, 2),
            "descripcion": info["descripcion"],
        })

    pct = top3[0]["porcentaje"]
    if pct >= 80:
        confianza, emoji = "Alta",  "🟢"
    elif pct >= 55:
        confianza, emoji = "Media", "🟡"
    else:
        confianza, emoji = "Baja",  "🔴"

    return {
        "prediccion_principal": {**top3[0], "confianza": confianza, "emoji": emoji},
        "top_3": top3,
    }


# ──────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path(__file__).parent / "index.html"
    if not p.exists():
        raise HTTPException(404, "index.html no encontrado")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tipos_ok = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in tipos_ok:
        raise HTTPException(400, f"Formato no soportado: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Archivo vacio")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Maximo 10 MB")

    return JSONResponse(predecir(data))


@app.get("/health")
async def health():
    ok = estado["model"] is not None
    return {
        "estado":         "ok" if ok else "sin_modelo",
        "modelo_cargado": ok,
        "total_clases":   len(estado["labels"]) if estado["labels"] else 0,
        "ruta_modelo":    MODEL_PATH,
    }


@app.get("/clases")
async def clases():
    if not estado["labels"]:
        return {"total": 0, "clases": []}
    resultado = []
    for idx, clave in estado["labels"].items():
        info = DESCRIPCIONES.get(clave, {"titulo": clave, "descripcion": ""})
        resultado.append({"id": int(idx), "clase": clave, **info})
    return {"total": len(resultado), "clases": resultado}
