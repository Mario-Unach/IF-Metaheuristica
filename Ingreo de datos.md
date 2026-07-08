# [cite_start]CRONOGRAMA DEL PROYECTO [cite: 1]
## [cite_start]Hibridación de Metaheurísticas con Redes Neuronales Artificiales (6 Semanas) [cite: 2]

### [cite_start]Objetivo General [cite: 3]
[cite_start]Diseñar, implementar y evaluar una arquitectura híbrida que combine una metaheurística (GWO optimización por lobo gris, MFO optimización por polilla y llama) con una red neuronal artificial (MLP o CNN), comparando su desempeño con un modelo base. [cite: 4]

| Semana | Actividades | Entregable |
| :--- | :--- | :--- |
| 1 | Formación de grupos. Selección del problema. Búsqueda del dataset. Diseño preliminar de la arquitectura neuronal. | Presentación corta (5 min) del problema y dataset seleccionado. |
| 2 | Implementación y entrenamiento del modelo base (MLP o CNN). Evaluación inicial de métricas. | Informe técnico del modelo base y resultados preliminares. |
| 3 | Investigación bibliográfica sobre hibridación. Selección de la metaheurística (GA, PSO, ACO, SA o TS). Diseño de la arquitectura híbrida. | Exposición de 5 minutos sobre la propuesta de hibridación. |
| 4 | Implementación de la metaheurística y su integración con la red neuronal. Pruebas iniciales del modelo híbrido. | Código funcional de la arquitectura híbrida. |
| 5 | Ejecución de experimentos. Ajuste de parámetros. Comparación entre modelo base y modelo híbrido. | Tablas comparativas y análisis de resultados. |
| 6 | Elaboración del informe final y presentación de resultados. Discusión de ventajas, limitaciones y posibles líneas futuras de investigación. | Exposición final (5 minutos) e informe completo del proyecto. |
[cite_start]*(Datos de la tabla [cite: 5])*

---

### [cite_start]Tablas Obligatorias [cite: 6]

[cite_start]**Tabla 1. Características del Dataset** [cite: 7]

| Característica | Valor |
| :--- | :--- |
| Nombre del Dataset | |
| Número de registros | |
| Número de variables | |
| Variable objetivo | |
| Tipo de problema | |
[cite_start]*(Datos de la tabla [cite: 8])*

[cite_start]**Tabla 2. Configuración de la Red Neuronal** [cite: 9]

| Parámetro | Valor |
| :--- | :--- |
| Tipo de Red | Es una red neuronal Feedforward (hacia adelante) o secuencial, construida con la API Sequential de Keras. |
| Capas | - Una capa de entrada (implícita, definida por la primera capa oculta).- Dos capas ocultas (Dense).- Una capa de salida (Dense). |
| Neuronas/Filtros | - Primera capa oculta: 32 neuronas.<br>- Segunda capa oculta: 16 neuronas.<br>- Capa de salida: 1 neurona. |
| Función de Activación | - Capas ocultas: ReLU (Rectified Linear Unit).<br>- Capa de salida: Sigmoid (para clasificación binaria). |
| Learning Rate | Se utiliza el optimizador Adam, que tiene un learning rate adaptativo por defecto (normalmente 0.001). |
| Épocas | El modelo fue entrenado durante 20 épocas (epochs=20). |
[cite_start]*(Datos de la tabla [cite: 10])*

[cite_start]**Tabla 3. Configuración de la Metaheurística** [cite: 11]

| Parámetro | Valor |
| :--- | :--- |
| Algoritmo | GA, PSO, ACO, SA o TS |
| Iteraciones | |
| Tamaño de población | |
| Parámetros específicos | |
[cite_start]*(Datos de la tabla [cite: 12])*

[cite_start]**Tabla 4. Comparación de Resultados** [cite: 13]

| Modelo | Accuracy | Precision | Recall | F1-Score | Tiempo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Modelo Base | 0.8218 | 0.83 | 0.96 | 0.89 | 4673 |
| Modelo Híbrido 1 | | | | | |
| Modelo Híbrido 2 | | | | | |
| Modelo Híbrido 3 | | | | | |
[cite_start]*(Datos de la tabla [cite: 14])*

---

### [cite_start]Producto Final [cite: 15]
[cite_start]Cada grupo deberá entregar: [cite: 16]
* [cite_start]Presentación final. [cite: 17]
* [cite_start]Código fuente. [cite: 18]
* [cite_start]Dataset utilizado. [cite: 19]
* [cite_start]Arquitectura propuesta. [cite: 20]
* [cite_start]Tablas comparativas. [cite: 21]
* [cite_start]Análisis de resultados. [cite: 22]
* [cite_start]Conclusiones. [cite: 23]
* [cite_start]Propuesta de artículo científico en formato IEEE. [cite: 24]