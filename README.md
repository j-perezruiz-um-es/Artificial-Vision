# Clasificación de Piezas Mecánicas mediante Deep Learning

**Proyecto Final - Visión Artificial | 4º Ingeniería | Universidad de Murcia**  
Autor: Jorge Pérez

---

## Descripción

Sistema de inspección automática de piezas mecánicas (tornillos) en una cinta transportadora industrial. El proyecto cubre todo el pipeline: desde la preparación del dataset hasta la inferencia, pasando por el dimensionamiento del hardware real del sistema de visión.

El clasificador alcanza una **precisión del 94%** sobre 13 tipos distintos de tornillos del dataset MVTec Screws.

---

## Estructura del Proyecto

```
Proyecto Final/
├── Codigo_ProyectoFinal_JorgePerez.ipynb   # Notebook principal
├── MemoriaProyectoVision.pdf               # Memoria técnica
├── MemoriaProyectoVision.docx              # Memoria técnica (editable)
├── PresentacionVision.pptx                 # Presentación
├── PresentacionProyectoVisionArtificial.pdf
└── README.md
```

---

## Pipeline

### Bloque 1 — Preparación del Dataset
- Dataset fuente: **MVTec Screws v1.0** (384 imágenes de escena completa con anotaciones JSON)
- Se recortan automáticamente las piezas individuales usando las bounding boxes del JSON
- Estrategia de recorte cuadrado adaptativo: `L = max(w, h) + 20px`
- Resultado: **4.427 imágenes** organizadas en 13 carpetas de clase

**Splits:**
| Conjunto | Imágenes |
|----------|----------|
| Train    | 3.098 (70%) |
| Val      | 664 (15%) |
| Test     | 665 (15%) |

### Bloque 2 — Entrenamiento del Modelo
- **Arquitectura**: ResNet50 pre-entrenado en ImageNet (transfer learning)
- Se congelan las primeras 49 capas; solo se entrena la capa `fc` final (13 clases)
- Entrada: imágenes redimensionadas a **224×224** con normalización ImageNet
- **Data Augmentation**: rotación ±15°, flip horizontal, jitter de brillo
- Optimizador: Adam (lr=0.001), CrossEntropyLoss, 10 épocas, batch=32
- Entrenamiento acelerado en **GPU (CUDA)**

### Bloque 3 — Dimensionamiento del Sistema de Visión
Cálculos teóricos para un despliegue industrial real:

| Parámetro | Valor |
|-----------|-------|
| Resolución espacial | 44.4 µm/píxel |
| Sensor | 5 MP — Sony IMX264 (Global Shutter) |
| Campo de visión | ~108 × 91 mm |
| Tiempo de exposición máx. | 44 µs (cinta a 1 m/s) |
| Distancia focal calculada | 48.8 mm → lente comercial 50 mm |
| Distancia de trabajo | 600 mm |

**Presupuesto hardware estimado: ~800 €**
- Cámara Basler ace 2 (a2A2448-75ucBAS): ~500 €
- Lente Computar M5028-MPW3 (50mm, 2/3"): ~180 €
- Anillo LED industrial: ~120 €

### Bloque 4 — Evaluación e Inferencia
- Accuracy global en test: **94%**
- El modelo se guarda en `modelo_tornillos_resnet50.pth`

**Resultados por clase (extracto):**

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| type_001 | 0.69 | 0.98 | 0.81 |
| type_006 | 0.98 | 0.97 | 0.97 |
| type_008 | 0.98 | 0.98 | 0.98 |
| type_013 | 0.96 | 1.00 | 0.98 |

> `type_001` tiene baja precisión (falsos positivos) y `type_003` tiene bajo recall — el resto del modelo es muy robusto.

---

## Requisitos

```
Python >= 3.8
torch
torchvision
tensorflow >= 2.18
opencv-python
numpy
pandas
scikit-learn
matplotlib
tqdm
Pillow
```

Instalar con:
```bash
pip install torch torchvision tensorflow opencv-python numpy pandas scikit-learn matplotlib tqdm Pillow
```

---

## Uso

1. Descargar el dataset **MVTec Screws v1.0** y colocarlo en la raíz del proyecto como `mvtec_screws_v1.0/`
2. Ejecutar el notebook `Codigo_ProyectoFinal_JorgePerez.ipynb` bloque a bloque
3. El modelo entrenado se guarda automáticamente como `modelo_tornillos_resnet50.pth`

---

## Dataset

[MVTec Screws Dataset](https://www.mvtec.com/company/research/datasets) — Paul Bergmann et al., MVTec Software GmbH.
