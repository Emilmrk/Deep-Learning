# Clasificación de Emociones con Deep Learning - FER2013

Este proyecto implementa y compara diferentes modelos de redes neuronales convolucionales (CNN) para la clasificación de emociones humanas usando el dataset **FER2013**. El objetivo es evaluar el desempeño de varios enfoques y etiquetar automáticamente un conjunto propio de imágenes no clasificadas.

---

## 📁 Estructura del Proyecto

El proyecto está organizado por carpetas, una por cada modelo:

```
inteligencia/
│
├── CNNBASE/         # Modelo CNN base
├── VGGBASE/         # Modelo basado en VGG16
├── RESNETBASE/      # Modelo basado en ResNet50 (fine-tuning)
├── MOBNETBASE/      # Modelo basado en MobileNetV2 (fine-tuning)
│
├── fotos/           # Imágenes propias sin clasificar
├── train/           # Dataset FER2013 (ya separado por clase)
│
├── final/           # Entorno virtual (ignorado en Git)
├── requirements.txt
├── .gitignore
└── README.md
```

Cada carpeta de modelo incluye:

- `train*.py`: script de entrenamiento
- `etiquetado*.py`: script de etiquetado automático
- `*.h5`: modelo entrenado
- `*.png`: gráficas de accuracy/loss y matriz de confusión
- `*.csv`: predicciones y conteo de clases

---

## 📦 Requisitos

Los paquetes necesarios están listados en `requirements.txt`.

### ✅ Crear entorno virtual e instalar dependencias:

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python3 -m venv final
source final/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Cómo entrenar un modelo

```bash
cd CNNBASE/
python3 traincnn.py
```

Repite el proceso para `trainvgg.py`, `trainresnet.py`, `trainmobnet.py`, etc., según el modelo que quieras ejecutar.

---

## 🧪 Cómo etiquetar imágenes propias

```bash
cd RESNETBASE/
python3 etiquetadoresnet.py
```

Esto generará:

- Las imágenes etiquetadas en subcarpetas
- Un archivo `.csv` con los resultados
- Una gráfica de conteo por clase

---

## 📊 Métricas generadas

Para cada modelo se generan automáticamente:

- Accuracy y loss por época (`*.png`)
- Matriz de confusión (`*.png`)
- Resultados por clase (`*.csv`)

---

## 📚 Dataset

Se utiliza el dataset FER2013 con imágenes de 48x48 píxeles en escala de grises, clasificadas en 7 emociones:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 👨‍💻 Autor

Desarrollado por Emilmrk como parte de un estudio de comparación de modelos CNN para clasificación de emociones con FER2013.

---
