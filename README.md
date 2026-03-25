# 214729 — Práctica 2 Unidad II

## Clasificador de Vehículos con Redes Neuronales Convolucionales

Modelo que identifica el tipo de vehículo en una imagen utilizando Transfer Learning con MobileNetV2 implementado con Keras y FastAPI.

Se recomienda clonar este repositorio y seguir las instrucciones para facilitar el uso del modelo y su interfaz.

---

## Clonar repositorio

```bash
git clone https://github.com/OscarAlonsoNavaRivera-214729/214729_Practica_2_UnidadII.git
```

O usando SSH:

```bash
git clone git@github.com:OscarAlonsoNavaRivera-214729/214729_Practica_2_UnidadII.git
```

---

## Requisitos

* Python 3.10 o superior
* pip

---

## Instalación

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

## Ejecutar la aplicación

```bash
fastapi dev main.py
```

Abrir en el navegador:

```
http://localhost:8000
```

### Estado del modelo

* Verde ("Modelo listo"): funcionamiento correcto
* Rojo: verificar que existan los archivos:

  * `modelo_vehiculos.keras`
  * `labels.json`

Ambos deben estar en la misma carpeta que `main.py`.

---

## Uso

1. Arrastrar una imagen o seleccionarla manualmente
2. Formatos soportados: JPG, PNG, WEBP (máximo 10 MB)
3. Presionar "Clasificar"
4. El sistema mostrará:

   * Tipo de vehículo detectado
   * Nivel de confianza
   * Tres predicciones más probables

---

## API

### Endpoint principal

```http
POST /predict
```

### Request

* Content-Type: multipart/form-data
* Parámetro: file (imagen)

### Response (ejemplo)

```json
{
  "prediccion": "suv",
  "confianza": 0.92,
  "top_3": [
    {"clase": "suv", "probabilidad": 0.92},
    {"clase": "pickup", "probabilidad": 0.05},
    {"clase": "van", "probabilidad": 0.03}
  ]
}
```

---

## Estructura del proyecto

```
app/
├── main.py                  # Servidor FastAPI y lógica de predicción
├── train_model.py           # Script para re-entrenar el modelo
├── index.html               # Interfaz web
├── requirements.txt         # Dependencias
├── modelo_vehiculos.keras   # Modelo entrenado
└── labels.json              # Clases del modelo
```

---

## Modelo

* Arquitectura: MobileNetV2 (preentrenado en ImageNet)
* Tipo: Clasificación multiclase
* Tamaño de entrada: 224x224
* Función de pérdida: categorical_crossentropy
* Optimizador: Adam
* Métrica: accuracy

---

## Resultados (referenciales)

* Accuracy entrenamiento: ~90%
* Accuracy validación: ~80–85%

Nota: los resultados pueden variar dependiendo del dataset utilizado.

---

## Re-entrenar el modelo (opcional)

El modelo incluido ya está entrenado. Solo es necesario re-entrenar si se agregan nuevas clases o imágenes.

### Estructura requerida

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

Se recomienda al menos 40 imágenes por clase (idealmente más).

### Ejecutar entrenamiento

```bash
python train_model.py
```

Esto generará un nuevo archivo:

```
modelo_vehiculos.keras
```

---

## Limitaciones

* Dataset reducido puede afectar la capacidad de generalización del modelo
* Sensible a condiciones de iluminación, ángulo y calidad de imagen
* Puede existir confusión entre clases visualmente similares

---

## Tecnologías

| Paquete            | Uso                         |
| ------------------ | --------------------------- |
| fastapi            | Servidor de la API REST     |
| keras + tensorflow | Red neuronal convolucional  |
| pillow             | Procesamiento de imágenes   |
| numpy              | Operaciones numéricas       |
| scikit-learn       | Utilidades de entrenamiento |

---

## Notas

* La interfaz está servida directamente desde FastAPI para simplificar el despliegue
* El modelo se carga en memoria al iniciar el servidor
* Se recomienda usar entorno virtual para evitar conflictos de dependencias

---

## Licencia

Uso académico y educativo.
