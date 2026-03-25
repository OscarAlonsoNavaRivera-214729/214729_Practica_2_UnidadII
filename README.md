# 214729 — Practica 2 Unidad II
## Clasificador de Vehiculos con Redes Neuronales Convolucionales

Modelo que identifica el tipo de vehiculo en una imagen usando Transfer Learning con MobileNetV2 (Keras + FastAPI).
---

## Requisitos

- Python 3.10 o superior
- pip

---

## Instalacion

```bash
# 1. Entrar a la carpeta del proyecto
cd 214729_Practica_2_UnidadII

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

## Correr la aplicacion

```bash
fastapi dev main.py
```

Abrir en el navegador: **http://localhost:8000**

El punto en la esquina superior derecha debe aparecer en verde con el texto **"Modelo listo"**. Si aparece en rojo, verificar que `modelo_vehiculos.keras` y `labels.json` esten en la misma carpeta que `main.py`.

---

## Uso

1. Arrastra una imagen al area de carga o haz clic para seleccionarla
2. Formatos aceptados: JPG, PNG, WEBP — maximo 10 MB
3. Presiona **Clasificar**
4. Se muestra el tipo de vehiculo detectado, el porcentaje de certeza y las 3 opciones mas probables

---

## Estructura del proyecto

```
app/
├── main.py                  servidor FastAPI y logica de prediccion
├── train_model.py           script para re-entrenar el modelo
├── index.html               interfaz web
├── requirements.txt         dependencias
├── modelo_vehiculos.keras   modelo ya entrenado
└── labels.json              clases del modelo
```

---

## Re-entrenar el modelo (opcional)

El modelo incluido ya esta entrenado. Solo es necesario re-entrenar si se agregan imagenes o clases nuevas.

Crear la carpeta `images/` con una subcarpeta por clase y al menos 40 imagenes cada una:

```
images/
├── sedan/
├── hatchback/
├── suv/
├── pickup/
├── van/
├── truck/
├── bus/
├── motorcycle/
└── bicycle/
```

```bash
python train_model.py
```

Esto genera un nuevo `modelo_vehiculos.keras` que reemplaza el anterior.

---

## Tecnologias

| Paquete | Uso |
|---|---|
| fastapi | Servidor de la API REST |
| keras + tensorflow | Red neuronal convolucional |
| pillow | Procesamiento de imagenes |
| numpy | Operaciones numericas |
| scikit-learn | Utilidades de entrenamiento |
