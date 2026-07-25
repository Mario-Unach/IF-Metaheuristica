# Optimización Híbrida GWO-MFO y Bayesiana de XGBoost para Predicción de Incumplimiento Crediticio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-green.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.9.0-orange.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este repositorio contiene una investigación comparativa exhaustiva sobre diferentes estrategias de optimización de hiperparámetros para **XGBoost** aplicado a la predicción de riesgo crediticio en datos desbalanceados. El proyecto contrasta el **estado del arte de la industria** (Optimización Bayesiana con Optuna) contra **algoritmos metaheurísticos bioinspirados** (Grey Wolf Optimizer, Moth-Flame Optimization y un híbrido GWO-MFO).

### 🎯 Objetivo Principal

Evaluar si los algoritmos de inteligencia de enjambre pueden superar o igualar el rendimiento de la optimización bayesiana en la tarea de detección de incumplimiento de pago en tarjetas de crédito, utilizando como métrica principal el **PR-AUC** (Precision-Recall Area Under the Curve) adecuada para datasets desbalanceados.

---

## 🗂️ Estructura del Repositorio

```
/workspace
├── Dataset/
│   ├── default of credit card clients.csv    # Dataset principal (30,000 registros)
│   └── default of credit card clients.xls    # Versión Excel del dataset
│
├── Notebooks/
│   └── XGBoost/
│       ├── XGBoost Base/                     # Modelo baseline sin optimización
│       │   └── XGBoost Base.ipynb
│       ├── XGBoost + Optuna/                 # Optimización Bayesiana (Benchmark)
│       │   ├── XGBoost + Optuna.ipynb
│       │   └── resultados_optuna/            # Resultados y visualizaciones
│       ├── XGBoost + GWO/                    # Grey Wolf Optimizer
│       │   ├── XGBoost + GWO.ipynb
│       │   └── Anexos/                       # Análisis y figuras complementarias
│       └── XGBoost + GWO + MFO/              # Híbrido GWO-MFO (Propuesta)
│           └── xgboost-gwo-mfo.ipynb
│
├── Documents/
│   ├── Documento latex/                      # Paper académico en LaTeX
│   │   ├── main.tex                          # Archivo principal del paper
│   │   ├── main.pdf                          # Versión compilada del paper
│   │   ├── resultados_gwo_mfo/               # Figuras para el paper
│   │   └── resultados_optuna/                # Figuras comparativas
│   ├── ideas.md                              # Documentación conceptual y análisis
│   ├── META_INFORME.docx                     # Informe técnico
│   ├── CRONOGRAMA DEL PROYECTO.docx          # Planificación temporal
│   └── Paper_Metaherustica.pdf               # Versión final del artículo
│
├── requirements.txt                          # Dependencias del proyecto
└── README.md                                 # Este archivo
```

---

## 🔬 Metodología

### Modelos Implementados

| Modelo | Tipo | Propósito |
|--------|------|-----------|
| **XGBoost Base** | Baseline | Punto de partida con parámetros por defecto |
| **XGBoost + Optuna** | Benchmark Industrial | Optimización Bayesiana (TPE) - Estado del arte |
| **XGBoost + GWO** | Metaheurística Pura | Grey Wolf Optimizer - Exploración global |
| **XGBoost + MFO** | Metaheurística Pura | Moth-Flame Optimization - Explotación local |
| **XGBoost + H-GWO-MFO** | **Propuesta Híbrida** | Combina exploración (GWO) + explotación (MFO) |

### Estrategia de Optimización Híbrida

El híbrido **GWO-MFO** implementa una estrategia de **cambio de fase**:

1. **Fase 1 (Iteraciones 1-N/2)**: Usa las ecuaciones de movimiento de **GWO** para exploración agresiva del espacio de hiperparámetros
2. **Fase 2 (Iteraciones N/2-N)**: Cambia a la espiral de **MFO** para afinamiento local alrededor de la mejor solución encontrada

### Métricas de Evaluación

Dado el desbalance del dataset (~78% clase mayoritaria "paga", ~22% clase minoritaria "incumple"), se priorizan:

- **PR-AUC** (Precision-Recall AUC) - Métrica principal
- **F1-Score** - Balance entre precisión y recall
- **ROC-AUC** - Capacidad discriminativa general
- **Recall (Sensibilidad)** - Detección de morosos
- **Matriz de Confusión** - Análisis detallado de errores

---

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip o gestor de paquetes compatible
- Jupyter Notebook / JupyterLab

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd <directorio-del-proyecto>

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

```txt
xgboost>=3.0.0
pandas
numpy
scikit-learn
matplotlib
seaborn
ipykernel
jupyter
optuna>=4.0.0
plotly
mealpy==3.0.3
scipy
```

### Ejecución de los Notebooks

Los notebooks están diseñados para ejecutarse secuencialmente:

1. **Baseline**: `Notebooks/XGBoost/XGBoost Base/XGBoost Base.ipynb`
2. **Benchmark**: `Notebooks/XGBoost/XGBoost + Optuna/XGBoost + Optuna.ipynb`
3. **Metaheurísticas**: `Notebooks/XGBoost/XGBoost + GWO/XGBoost + GWO.ipynb`
4. **Híbrido**: `Notebooks/XGBoost/XGBoost + GWO + MFO/xgboost-gwo-mfo.ipynb`

Cada notebook genera automáticamente sus resultados en carpetas locales (`resultados_*`).

---

## 📊 Dataset

Se utiliza el dataset **"Default of Credit Card Clients"** que contiene:

- **30,000 registros** de clientes de tarjeta de crédito
- **23 características** incluyendo:
  - Límite de crédito (`LIMIT_BAL`)
  - Demografía (sexo, educación, estado civil, edad)
  - Historial de pagos (`PAY_0` a `PAY_6`)
  - Montos facturados (`BILL_AMT1` a `BILL_AMT6`)
  - Montos pagados (`PAY_AMT1` a `PAY_AMT6`)
- **Variable objetivo**: `default payment next month` (binaria: 0=paga, 1=incumple)

**Distribución de clases:**
- Clase 0 (Paga): ~78%
- Clase 1 (Incumple): ~22%

---

## 📈 Resultados Principales

### Hallazgos Clave

1. **XGBoost Base** confirma que el algoritmo puede aprender patrones de riesgo crediticio, pero sufre de sobreajuste y mala calibración sin optimización.

2. **Optuna** logra convergencia rápida a un PR-AUC superior en ~30 trials, demostrando ser ideal para despliegues en producción por su eficiencia computacional.

3. **GWO y MFO puros** muestran capacidades complementarias:
   - GWO: Excelente exploración global, evita óptimos locales
   - MFO: Refinamiento local preciso mediante espirales logarítmicas

4. **Híbrido GWO-MFO** alcanza en **20 iteraciones** lo que a las metaheurísticas puras les toma 50, demostrando equilibrio óptimo entre exploración y explotación.

### Visualizaciones Generadas

Cada notebook produce:
- Curvas de convergencia (Iteración vs PR-AUC)
- Boxplots de robustez (15 ejecuciones independientes)
- Matrices de confusión
- Curvas ROC y Precision-Recall
- Tablas comparativas de métricas

---

## 📝 Publicación Académica

Este trabajo está documentado en un paper académico formato **IEEE Latin America Transactions** ubicado en:

```
Documents/Documento latex/main.tex
```

El paper incluye:
- Revisión bibliográfica del estado del arte
- Formulación matemática de GWO, MFO y el híbrido
- Diseño experimental riguroso
- Análisis estadístico (test de Wilcoxon)
- Discusión de resultados y conclusiones

---

## 👥 Autores

| Autor | Afiliación | Contacto |
|-------|------------|----------|
| **Lenin Francisco López Cabrera** | Universidad Nacional de Chimborazo, Ecuador | leninf.lopez@unach.edu.ec |
| **Jose Mario Camacho Monar** | Universidad Nacional de Chimborazo, Ecuador | - |

**Facultad de Ingeniería**  
**Ciencia de Datos e Inteligencia Artificial**  
**Universidad Nacional de Chimborazo**  
Riobamba, Ecuador

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Librería **MealPy** por la implementación de algoritmos metaheurísticos
- **Optuna** por el framework de optimización bayesiana
- Comunidad de XGBoost por el algoritmo base
- Universidad Nacional de Chimborazo por el apoyo institucional

---

## 📚 Referencias Clave

1. Grinsztajn, L., et al. (2022). "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS*.
2. Mirjalili, S., et al. (2014). "Grey Wolf Optimizer." *Advances in Engineering Software*.
3. Mirjalili, S. (2015). "Moth-flame optimization algorithm." *Knowledge-Based Systems*.
4. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*.

---

## 🔗 Enlaces de Interés

- [Documentación de XGBoost](https://xgboost.readthedocs.io/)
- [Documentación de Optuna](https://optuna.readthedocs.io/)
- [Repositorio de MealPy](https://github.com/mealpy/mealpy)
- [Dataset UCI Credit Default](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

<div align="center">

**Si este proyecto te fue útil, considera darle una ⭐️**

Hecho con ❤️ para la comunidad de Ciencia de Datos

</div>
