# AdM TP — Predicción de ACV con Ensamble de Clasificadores

**Materia:** Aprendizaje de Máquina I  
**Facultad:** Ingeniería — Universidad de Buenos Aires  
**Dataset:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

**Integrantes :**
-  Capón Paul, Lucia Tamara
- Madrid, Martin
- Orellana, César Andrés
- Britez, Leandro

---

## Descripción del proyecto

Este trabajo práctico investiga la predicción de accidentes cerebrovasculares (ACV/stroke) a partir de variables clínicas y demográficas de pacientes. El dataset contiene ~5.000 registros con un desbalance severo de clases (95% sin ACV / 5% con ACV), lo que representa el desafío central del problema.

**Hipótesis central:** La combinación de modelos con distintos sesgos inductivos (árboles simétricos, boosting por gradiente, máquinas de vectores) superará a cualquier modelo individual en la detección de la clase minoritaria (stroke=1).


## Estructura del proyecto

```
AdM_TP/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Dataset original
│
├── notebooks/
│   ├── 00_eda.ipynb                          # Análisis exploratorio compartido
│   ├── 00_model_Baseline.ipynb               # Heurística baseline (reglas del EDA)
│   ├── 01_model_A.ipynb                      # Random Forest 
│   ├── 02_model_B.ipynb                      # SVM
│   ├── 03_model_C.ipynb                      # CatBoost + GMM Feature Eng. 
│   ├── 04_model_D.ipynb                      # XGBoost 
│   └── 05_ensemble.ipynb                     # Ensamble: Hard / Soft / Weighted / Stacking
│
├── src/
│   ├── data_loader.py                        # Carga, limpieza y split (fuente única de verdad)
│   ├── evaluation.py                         # Métricas y visualizaciones compartidas
│   └── model_registry.py                     # Guardar y cargar modelos entrenados
│
├── artifacts/
│   ├── model_A/                              # Random Forest entrenado + params.yaml
│   ├── model_B/                              # SVM entrenado + params.yaml
│   ├── model_C/                              # CatBoost entrenado + params.yaml
│   └── model_D/                              # XGBoost entrenado + params.yaml
│
├── reports/                                  # Reportes y gráficos finales
├── pyproject.toml                            # Dependencias del proyecto
└── README.md
```

---

## Instalación y setup

### Requisitos previos

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- En Mac: `brew install libomp` (requerido por XGBoost)

### Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd AdM_TP

# Instalar dependencias base (incluye CatBoost y Optuna)
uv sync

# Instalar dependencias opcionales (XGBoost y LightGBM)
uv sync --extra boost
```

### En caso de error con XGBoost en Mac (Apple Silicon)

```bash
brew install libomp
uv cache clean
rm -rf .venv
uv sync --extra boost
```

---

## Flujo de trabajo

### Regla fundamental

> **Todos los integrantes deben usar `src.data_loader.get_train_test_split()`**  
> Esto garantiza que el train/test split sea **idéntico** para todos (seed fija = 13, stratify=y).  
> Nunca hacer el split manualmente en los notebooks.

### Orden de ejecución

```
1. 00_eda.ipynb              → Exploración compartida, análisis del desbalance
2. 00_model_Baseline.ipynb   → Heurística de referencia
3. 01_model_A.ipynb          → Entrenar y guardar Model A
4. 02_model_B.ipynb          → Entrenar y guardar Model B
5. 03_model_C.ipynb          → Entrenar y guardar Model C
6. 04_model_D.ipynb          → Entrenar y guardar Model D
7. 05_ensemble.ipynb         → Cargar los 4 modelos y evaluar ensambles
```

> **Importante:** cada notebook de modelo debe terminar con `save_model(...)` para que el ensamble pueda cargarlo desde `artifacts/`.

---

## Enfoque metodológico

El problema de predicción de ACV se caracteriza por un alto desbalance de clases (~5% positivos) y un contexto donde el costo de error es asimétrico: no detectar un ACV (falso negativo) es significativamente más grave que generar una falsa alarma.

En este contexto, el objetivo del proyecto no es maximizar la accuracy, sino optimizar la detección de la clase minoritaria, manteniendo un equilibrio razonable con la precisión.

### Baseline

Antes de aplicar modelos de machine learning, se construyó una heurística basada en el EDA, utilizando variables con mayor asociación al target:

Edad > 60
Hipertensión
Enfermedad cardíaca
Glucosa elevada (>140)

Esta heurística permite:
- establecer un piso mínimo de performance
- contar con un modelo completamente interpretable

A pesar de su simplicidad, el baseline logra capturar parcialmente el riesgo (recall moderado), lo que indica que gran parte de la señal del problema es accesible mediante reglas simples.

## Modelos individuales 

Se seleccionaron modelos con enfoques complementarios:

Random Forest (Model A): captura interacciones no lineales y es robusto al ruido
SVM (Model B): introduce un enfoque geométrico basado en márgenes
CatBoost + GMM (Model C): incorpora feature engineering no supervisado
XGBoost (Model D): modelo de boosting altamente expresivo

La intención no es encontrar “el mejor modelo único”, sino construir diversidad, condición clave para un ensamble efectivo.


## Descripción de los modelos

### Baseline — Heurística (reglas del EDA)
Modelo de referencia basado en reglas simples derivadas del análisis exploratorio. Predice stroke=1 si el paciente cumple al menos 2 de las siguientes condiciones: edad > 60, hipertensión, enfermedad cardíaca, glucosa > 140. Establece el piso mínimo de comparación para los modelos reales.

### Model A — Random Forest 
Ensemble de árboles de decisión con `class_weight='balanced'` para manejar el desbalance. Preprocesamiento con `ColumnTransformer` (imputación mediana + OHE). Optimización de hiperparámetros con Optuna sobre dos métricas: Recall y F1. Incluye análisis de threshold para maximizar detección de la clase positiva.

### Model B — SVM
Support Vector Machine con kernel RBF. Requiere escalado obligatorio de features (`StandardScaler`). Manejo del desbalance mediante `class_weight='balanced'`. Búsqueda de hiperparámetros con GridSearchCV sobre `C` y `gamma`. Sirve como contraste teórico al ser el único modelo no basado en árboles del ensamble.

### Model C — CatBoost + GMM Feature Engineering 
CatBoost con manejo nativo de variables categóricas mediante Ordered Target Statistics, evitando data leakage en el encoding. Feature engineering no supervisado con Gaussian Mixture Model (GMM) sobre las variables clínicas continuas (`age`, `avg_glucose_level`, `bmi`) para capturar subpoblaciones latentes. Se comparan tres variantes: Baseline (sin GMM), GMM Hard (etiqueta de cluster) y GMM Soft (probabilidades de pertenencia). Optimización con Optuna (TPE). Referencia: Prokhorenkova et al., NeurIPS 2018.


### Model D — XGBoost 
Gradient boosting con `scale_pos_weight` ajustado al ratio de desbalance (~19:1). Preprocesamiento con pipeline sklearn (imputación + escalado + OHE). Búsqueda de hiperparámetros con GridSearchCV sobre `subsample`, `learning_rate` y `scale_pos_weight`. Evaluado con AUC-ROC como métrica principal.

---
## Selección de métricas 

Dado el desbalance:

- Accuracy fue descartada por ser engañosa
- Recall (clase 1) se prioriza por el costo clínico de falsos negativos
- F1-score se utiliza como métrica de optimización para balancear recall y precisión 

Se optó por F1 como criterio principal, ya que evita soluciones extremas donde el modelo detecta casi todos los casos positivos a costa de una gran cantidad de falsas alarmas.

---
## Ensamble 

El ensamble final se construye utilizando los modelos A, B y D, mientras que el Modelo C se evalúa por separado debido a su enfoque distinto (feature engineering con GMM).

La hipótesis es que:

modelos con distintos errores y representaciones del problema pueden complementarse, mejorando la capacidad de detección de la clase minoritaria.

Se evalúan distintas estrategias:

Hard Voting
Soft Voting 
Stacking

## Estrategias de ensamble (`05_ensemble.ipynb`)

| Estrategia | Descripción | Cuándo es mejor |
|---|---|---|
| **Hard Voting** | Voto de mayoría sobre clases predichas | Modelos con distinto tipo de error |
| **Soft Voting** | Promedio de probabilidades | Modelos bien calibrados — **recomendado** |
| **Stacking** | Meta-learner (LogReg) sobre predicciones base | Máximo poder predictivo |

La métrica de evaluación principal del ensamble es **AUC-ROC** y **Recall de la clase positiva (stroke=1)**, ya que en contexto médico minimizar los falsos negativos (ACV no detectados) es prioritario sobre minimizar los falsos positivos.

---

## Métricas utilizadas

Dado el severo desbalance de clases (95%/5%), la **accuracy fue descartada** como métrica principal. Un clasificador trivial que predice siempre "No Stroke" obtiene ~95% de accuracy con Recall = 0%.

| Métrica | Descripción | 
|---|---|
| **AUC-ROC** | Capacidad discriminante a todos los umbrales |
| **Recall (clase 1)** | % de ACV reales detectados | 
| **F1-Score** | Balance Precision/Recall |  
| **Avg. Precision** | Área bajo curva Precision-Recall | 
| Accuracy | Descartada por ser engañosa con desbalance | 

---

## Resultado y conclusión

| Modelo                          | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------------------|----------|-----------|--------|----------|---------|
| Baseline (Heurística)          | 0.86     | 0.16      | 0.44   | 0.24     | -       |
| Ensamble (Hard Voting)         | 0.866    | 0.20      | 0.60   | 0.30     | -       |
| Model C (CatBoost + GMM)       | 0.755    | 0.153     | 0.88   | 0.26     | 0.886   |

### Baseline

| Clase | Precision | Recall | F1-score | Support |
|------|----------|--------|----------|---------|
| 0    | 0.97     | 0.88   | 0.92     | 972     |
| 1    | 0.16     | 0.44   | 0.24     | 50      |

### Detalle Ensamble (Hard Voting)

| Clase | Precision | Recall | F1-score |
|------|----------|--------|----------|
| 1    | 0.20     | 0.60   | 0.30     |

### Detalle Model C

| Métrica   | Valor  |
|----------|--------|
| Accuracy | 0.755  |
| Precision| 0.153  |
| Recall   | 0.88   |
| F1-score | 0.26   |
| ROC-AUC  | 0.886  |


En un contexto médico, donde el costo de no detectar un ACV es alto, modelos con alto recall son valiosos. Sin embargo, un sistema útil en la práctica debe balancear sensibilidad y precisión para evitar una cantidad excesiva de falsos positivos.

En este sentido, el ensamble se posiciona como la mejor solución global, al ofrecer una mejora clara sobre el baseline y un compromiso más razonable que el Modelo C entre detección y confiabilidad.

El ensamble por votación mayoritaria (hard voting) mejora la capacidad de detección respecto al baseline (recall = 0.60, f1 = 0.30), manteniendo una precisión moderada. Esto confirma que la combinación de modelos con distintos sesgos inductivos permite capturar patrones más complejos y generalizar mejor que una heurística fija.

Por otro lado, el Modelo C (CatBoost + GMM) presenta el mayor recall (0.88) y AUC-ROC (0.886), lo que indica una excelente capacidad para identificar casos positivos. Sin embargo, esto se logra a costa de una baja precisión (0.15), generando una alta tasa de falsos positivos y reduciendo la utilidad práctica del modelo en contextos donde las falsas alarmas deben ser controladas.

---

## Utilidades compartidas (`src/`)

### `data_loader.py`
```python
from src.data_loader import load_dataset, clean_data_id_gender, get_train_test_split, impute_bmi_knn

df = load_dataset()                          # Carga el CSV
df = clean_data_id_gender(df)               # Elimina 'id' y el registro gender='Other'
X_train, X_test, y_train, y_test = get_train_test_split(df)  # Split 80/20, seed=13, stratify
X_train, X_test = impute_bmi_knn(X_train, X_test)  # KNN imputer para BMI (fit solo en train)
```

### `evaluation.py`
```python
from src.evaluation import evaluate_model, plot_full_evaluation, compare_models

metrics = evaluate_model(model, X_test, y_test, model_name="Mi Modelo")
plot_full_evaluation(model, X_test, y_test, model_name="Mi Modelo")  # CM + ROC
df_comp = compare_models([metrics_A, metrics_B, metrics_C, metrics_D])
```

### `model_registry.py`
```python
from src.model_registry import save_model, load_model, load_all_models

# Guardar modelo entrenado
save_model(model=best_model, name="model_A", params=best_params, cv_score=0.87)

# Cargar en el notebook de ensamble
models = load_all_models()   # dict: {"model_A": estimator, "model_B": ...}
```

---

## Dependencias principales

| Paquete | Versión | Uso |
|---|---|---|
| scikit-learn | latest | Pipelines, métricas, SVM, RF, VotingClassifier |
| catboost | >= 1.2.0 | Model C |
| xgboost | latest | Model B (opcional: `uv sync --extra boost`) |
| optuna | >= 4.6.0 | Búsqueda bayesiana de HP (Model A y C) |
| pandas / numpy | latest | Manipulación de datos |
| matplotlib / seaborn | latest | Visualizaciones |
| joblib / pyyaml | latest | Persistencia de modelos |

---

## Fuente del dataset

**fedesoriano. (2021).** *Stroke Prediction Dataset.* Kaggle.  
 https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
