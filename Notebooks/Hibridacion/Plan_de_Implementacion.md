# Plan de Implementación: Hibridación Triple (XGBoost + GWO + MFO)

Este plan describe los pasos para implementar un modelo XGBoost optimizado mediante una hibridación secuencial de las metaheurísticas Lobo Gris (GWO) y Polilla-Llama (MFO), utilizando la librería `mealpy`, y culminando con un análisis estructurado.

## Propuesta de Hibridación (Relevo Secuencial)
Para mantener la simplicidad y el poder de `mealpy`, utilizaremos un enfoque de **Hibridación en Relevo**:
1. **Fase de Exploración (GWO):** El Lobo Gris se encargará de explorar todo el espacio de búsqueda globalmente durante la primera mitad de las iteraciones.
2. **Transferencia de Conocimiento:** Extraeremos la mejor solución y la distribución de la manada de lobos encontrada por GWO.
3. **Fase de Explotación (MFO):** Reduciremos (ajustaremos) los límites de búsqueda alrededor de la mejor solución de GWO y lanzaremos el algoritmo de la Polilla y la Llama para hacer una búsqueda microscópica fina (espiral logarítmica) y encontrar el hiperparámetro óptimo absoluto durante la segunda mitad de las iteraciones.

## Pasos de Ejecución

### 1. Creación y Estructuración del Notebook
- Archivo: `Notebooks/Hibridacion/XGBOOST+GWO+MFO.ipynb`.
- Todo documentado en español con formato Markdown:
  - Título Principal y Resumen teórico.
  - Explicación de cada celda de código antes de ejecutarla.

### 2. Preparación de Datos y Función Objetivo
- Carga de datos (rutas dinámicas relativas `../../Dataset/...`).
- Función objetivo multicriterio basada en el código base (maximizando PR-AUC y F1-Score con penalización por Recall bajo).

### 3. Implementación de la Hibridación con Mealpy
- Escribir la lógica de relevo automático.
- Experimentación 1: Ejecución Híbrida con **20 Iteraciones** (10 GWO + 10 MFO).
- Experimentación 2: Ejecución Híbrida con **50 Iteraciones** (25 GWO + 25 MFO).

### 4. Evaluación de Resultados y Gráficos
- Matrices de confusión.
- Curvas ROC y PR-AUC interactivas (Matplotlib/Plotly).
- Gráfico de convergencia comparativa.

### 5. Documento de Análisis (Paper .md)
- Archivo `Analisis_Hibridacion.md` estructurado como un paper científico.
- Incluirá metodología, resumen de los parámetros óptimos encontrados, imágenes generadas y conclusión final.
