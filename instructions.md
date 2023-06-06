# Instrucciones para el proyecto de Machine Learning

El notebook del proyecto cuenta con 8 apartados principales. Cada uno de estos apartados se encuentra en una sección distinta del notebook. A continuación se detallan los apartados y las secciones correspondientes:

### 1. Introducción

Introducción al proyecto y a la temática del mismo.

### 2. Datos

Análisis exploratorio de los datos y de las variables del dataset. Expliación de las variables y de su significado. Terminología utilizada en el proyecto.

### 3. Preprocesamiento

Preprocesamiento de los datos. Eliminación de variables, transformación de variables categóricas, normalización de variables numéricas, etc.

### 4. Modelos

Entrenamiento de los modelos de clasificación. Uso de modelos distintos para la clasificación de los datos. Comparación de los resultados obtenidos. Modelos discriminativos y modelos generativos.

### 5. Evaluación

Evaluación de los modelos. Uso de distintas métricas para la evaluación de los modelos. Comparación de los resultados obtenidos.

### 6. Re-modelado

Apartado descartado, como se explica en el reporte del proyecto.

Re-modelado de los datos. Uso de técnicas de re-modelado para mejorar los resultados obtenidos. Uso de técnicas de re-modelado para mejorar el rendimiento de los modelos. 

### 7. Optimización de hiperparámetros

Optimización de los hiperparámetros de los modelos. Uso de técnicas de optimización de hiperparámetros para mejorar los resultados obtenidos.

### Validación test

Validación del modelo candidato. Uso de técnicas de validación para comprobar la generalización del modelo candidato.


Para la replicación de los resultados del proyecto se debe tener en cuenta los siguientes puntos:

## Librerías utilizadas

1. Se deben tener instaladas las librerías de Python y las correspondientes versiones que se encuentran al inicio del notebook. En caso contrario, podría haber problemas de compatibilidad en algunas funciones o métodos.

```python
!pip install pandas==1.5.3
!pip install numpy==1.24.2
!pip install seaborn==0.12.2
!pip install sklearn==1.0.2
!pip install matplotlib==3.5.1
!pip install plotly==5.6.0
```

## Dataset

2. Se debe tener descargado el dataset `league_dataset.csv` en la misma carpeta en donde se encuetra el notebook del proyecto. Este dataset se puede encontrar en el siguiente enlace: [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

## Ejecución del notebook

3. Cumplidos los requisitos de compatibilidad, solo queda ejecutar el código. Para ello, se recomienda ejecutar el notebook en un entorno de desarrollo como Visual Studio Code, Jupyter Notebook o Google Colab. Se recomienda ejecutar siempre todo el preprocessing y luego dirigirse a los apartados de interés. Si se desea volver a ejecutar alguna celda, se recomienda repetir el proceso de ejecución desde el principio.

## Funciones auxiliares

4. Se debe tener descargadas las distintas funciones auxiliares utilizadas en el notebook. Estas funciones se encuentran en la carpeta `Resampling`. 

### `resampling.py`

La función `model_performance` calcula el rendimiento de un modelo utilizando diferentes métodos de remuestreo (error de entrenamiento, validación simple, validación cruzada de Monte Carlo, validación cruzada k-fold). Toma como entrada los siguientes parámetros:

- `library`: la biblioteca utilizada para construir el modelo (por ejemplo, sklearn, statsmodels, etc.).
- `method`: el método del modelo que se utilizará para el ajuste y la predicción.
- `X`: un DataFrame de pandas que contiene las características de los datos.
- `y`: una Serie de pandas que contiene la variable objetivo.
- `repeats` (opcional): el número de repeticiones del experimento (por defecto es 10).
- `k` (opcional): el número de divisiones en la validación cruzada k-fold (por defecto es 20).
- `model` (opcional): el nombre del modelo que se desea devolver (por defecto es "single").
- `rand` (opcional): la semilla aleatoria utilizada en el proceso (por defecto es 88).

La función calcula el rendimiento del modelo utilizando diferentes métodos de remuestreo y almacena los resultados en un DataFrame llamado `results_df` con las siguientes columnas: 'Resampling', 'Accuracy', 'Precision Macro', 'Recall Macro' y 'F1 Macro'.

A continuación, se describen los pasos principales de la función:

1. Entrenamiento del modelo con todo el conjunto de datos de entrenamiento y cálculo del error de entrenamiento.
2. Validación simple: se divide el conjunto de datos en conjuntos de entrenamiento y validación, se entrena el modelo en el conjunto de entrenamiento y se calcula el rendimiento en el conjunto de validación.
3. Validación cruzada de Monte Carlo: se realiza una validación cruzada de Monte Carlo con un número determinado de repeticiones y se calcula el rendimiento promedio del modelo.
4. Validación cruzada k-fold: se realiza una validación cruzada k-fold con un número determinado de divisiones y se calcula el rendimiento promedio del modelo.
5. Dependiendo del valor de `model`, la función devuelve el DataFrame `results_df` junto con el modelo entrenado correspondiente.

### `monte_carlo.py`


La función `monte_carlo_cv` implementa la validación cruzada de Monte Carlo para evaluar el rendimiento de un modelo. Esta técnica consiste en realizar múltiples divisiones aleatorias del conjunto de datos en conjuntos de entrenamiento y validación, ajustar el modelo en el conjunto de entrenamiento y evaluar su rendimiento en el conjunto de validación. La función toma los siguientes parámetros:

- `library`: la biblioteca utilizada para construir el modelo (por ejemplo, sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble).
- `method`: el método específico del modelo que se utilizará (por ejemplo, GaussianNB, DecisionTreeClassifier, RandomForestClassifier).
- `X`: las características del conjunto de datos.
- `y`: la variable objetivo del conjunto de datos.
- `n`: el número de iteraciones, es decir, la cantidad de divisiones aleatorias a realizar.
- `rand` (opcional): la semilla aleatoria utilizada en el proceso (por defecto es 88).

Dentro de la función, se realiza el siguiente procedimiento:

1. Se inicializan listas vacías para almacenar las métricas de rendimiento (exactitud, precisión, recall y puntuación F1) en cada iteración.
2. Se ejecuta un bucle `for` que realiza `n` iteraciones.
3. En cada iteración, se divide el conjunto de datos en conjuntos de entrenamiento y validación utilizando `train_test_split`, con una proporción de prueba del 33% y la semilla aleatoria especificada.
4. Se crea una instancia del modelo especificado mediante `method`.
5. Se ajusta el modelo en el conjunto de entrenamiento utilizando `fit`.
6. Se calculan las métricas de rendimiento (exactitud, precisión, recall y puntuación F1) comparando las etiquetas reales del conjunto de validación (`y_val`) con las predicciones del modelo en el conjunto de validación (`model.predict(X_val)`).
7. Las métricas de rendimiento de cada iteración se agregan a las listas correspondientes (`acc`, `prec`, `rec`, `f1`).
8. Al finalizar las iteraciones, se calcula el promedio de las métricas de rendimiento utilizando `np.mean`.
9. Además de las métricas de rendimiento, la función devuelve el modelo entrenado en la última iteración.

### `k_fold.py`

La función `k_fold_cv` implementa la validación cruzada k-fold para evaluar el rendimiento de un modelo. Esta técnica divide el conjunto de datos en k pliegues (folds), utiliza k-1 pliegues para entrenar el modelo y evalúa su rendimiento en el pliegue restante. La función toma los siguientes parámetros:

- `library`: la biblioteca utilizada para construir el modelo (por ejemplo, sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble).
- `method`: el método específico del modelo que se utilizará (por ejemplo, GaussianNB, DecisionTreeClassifier, RandomForestClassifier).
- `X`: las características del conjunto de datos.
- `y`: la variable objetivo del conjunto de datos.
- `k`: el número de pliegues en la validación cruzada k-fold.
- `rand` (opcional): la semilla aleatoria utilizada en el proceso (por defecto es 88).

Dentro de la función, se realiza el siguiente procedimiento:

1. Se crea una instancia de `KFold` con el número de pliegues especificado (`n_splits=k`), la opción de mezclar los datos (`shuffle=True`) y la semilla aleatoria (`random_state=rand`).
2. Se inicializan listas vacías para almacenar las métricas de rendimiento (exactitud, precisión, recall y puntuación F1) en cada pliegue.
3. Se ejecuta un bucle `for` que itera sobre los índices de los pliegues generados por `KFold.split(X)`.
4. En cada iteración, se divide el conjunto de datos en conjuntos de entrenamiento y validación utilizando los índices del pliegue actual.
5. Se crea una versión numérica de los conjuntos de entrenamiento y validación (`X_learn`, `y_learn`, `X_val`, `y_val`) para asegurarse de que sean matrices NumPy.
6. Se crea una instancia del modelo especificado mediante `method`.
7. Se ajusta el modelo en el conjunto de entrenamiento utilizando `fit`.
8. Se calculan las métricas de rendimiento (exactitud, precisión, recall y puntuación F1) comparando las etiquetas reales del conjunto de validación (`y_val`) con las predicciones del modelo en el conjunto de validación (`model.predict(X_val)`).
9. Las métricas de rendimiento de cada pliegue se agregan a las listas correspondientes (`acc`, `prec`, `rec`, `f1`).
10. Al finalizar los pliegues, se calcula el promedio de las métricas de rendimiento utilizando `np.mean`.
11. Además de las métricas de rendimiento, la función devuelve el modelo entrenado en el último pliegue.

En resumen, la función `k_fold_cv` realiza la validación cruzada k-fold para evaluar el rendimiento de un modelo utilizando divisiones específicas del conjunto de datos. Retorna las métricas de rendimiento promedio y el modelo entrenado en el último pliegue.