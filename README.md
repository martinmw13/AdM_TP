# AdM TP — Predicción de ACV con Ensamble de Clasificadores

**Materia:** Aprendizaje de Máquina I  
**Facultad:** Ingeniería — Universidad de Buenos Aires  
**Dataset:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

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

## Estrategias de ensamble (`05_ensemble.ipynb`)

| Estrategia | Descripción | Cuándo es mejor |
|---|---|---|
| **Hard Voting** | Voto de mayoría sobre clases predichas | Modelos con distinto tipo de error |
| **Soft Voting** | Promedio de probabilidades | Modelos bien calibrados — **recomendado** |
| **Weighted Voting** | Soft voting ponderado por AUC de CV | Cuando hay diferencia clara de calidad |
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
