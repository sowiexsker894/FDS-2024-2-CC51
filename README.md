# FDS-2024-2-CC51

Una consultora internacional, con sede en Lima, solicita desarrollar un proyecto de analítica con la finalidad de conocer las tendencias de los videos de YouTube en siete importantes países. El proyecto responde a la necesidad de su cliente, una importante empresa de marketing digital, que desea obtener respuestas a varios requerimientos .

# Análisis Predictivo: Factores que Influyen en el Número de Vistas en Videos

Este proyecto explora un conjunto de datos extenso y diverso para predecir el número de vistas en videos utilizando técnicas de aprendizaje automático. Se analizan múltiples variables y se evalúan los resultados obtenidos a partir de modelos de regresión.

## Contenidos

1. [INTRODUCCIÓN](#introduccion)

2. [OBJETIVOS](#objetivos)

3. [CONJUNTO DE DATOS](#datos)

4. [METODOLOGÍA](#metodologia)

5. [MODELO DE REGRESIÓN](#modelo)

6. [RESULTADOS](#resultados)

7. [CONCLUSIONES](#conclusiones)

8. [RECOMENDACIONES](#recomendaciones)

9. [BIBLIOGRAFÍA](#bibliografia)



---



## 1. Introducción <a name="introduccion"></a>



Este proyecto analiza un extenso conjunto de datos que incluye información detallada sobre videos publicados en una plataforma digital. Se busca identificar patrones y relaciones entre variables clave para predecir el número de vistas que puede alcanzar un video.



El análisis incluye:

- Procesamiento y limpieza de datos.

- Modelado predictivo utilizando técnicas avanzadas.

- Evaluación del modelo mediante métricas como RMSE, MAE y R².



---



## 2. Objetivos <a name="objetivos"></a>



### Objetivo General:

- Predecir el número de vistas de un video en función de diversas características.



### Objetivos Específicos:

- Realizar un análisis exploratorio para identificar las relaciones entre variables.

- Entrenar un modelo de regresión para realizar predicciones sobre el número de vistas.

- Evaluar el desempeño del modelo utilizando métricas estándar.



---



## 3. Conjunto de Datos <a name="datos"></a>



El conjunto de datos utilizado contiene miles de registros con variables relevantes. Las principales son:



| Variable        | Tipo    | Descripción                                 |

|------------------------|-------------|-----------------------------------------------------------------------------|

| `video_id`       | Texto    | Identificador único del video.                       |

| `title`        | Texto    | Título del video.                              |

| `channel_title`    | Texto    | Nombre del canal que publicó el video.                   |

| `category_id`     | Categórica | Categoría del video, representada por un ID numérico.            |

| `views`        | Numérica  | Número total de vistas del video.                      |

| `likes`        | Numérica  | Número de "Me gusta" recibidos.                       |

| `dislikes`       | Numérica  | Número de "No me gusta" recibidos.                     |

| `comment_count`    | Numérica  | Número total de comentarios.                        |

| `published_at`     | Fecha    | Fecha y hora de publicación del video.                   |

| `tags`         | Texto    | Etiquetas asociadas al video.                        |

| `ratings_disabled`   | Binaria   | Indica si las calificaciones están deshabilitadas.             |

| `comments_disabled`  | Binaria   | Indica si los comentarios están deshabilitados.               |

| `description`     | Texto    | Descripción proporcionada por el creador del video.             |



El conjunto de datos presenta alta dimensionalidad y heterogeneidad, lo que lo convierte en un caso interesante para aplicar técnicas avanzadas de aprendizaje automático.



---



## 4. Metodología <a name="metodologia"></a>



### Pasos:

1. **Análisis Exploratorio de Datos (EDA):**

  - Identificación de patrones generales en los datos.

  - Detección de valores atípicos y datos faltantes.



2. **Preprocesamiento:**

  - Limpieza de datos.

  - Transformación de variables categóricas y texto en valores numéricos.



3. **Selección de Variables:**

  - Variables predictoras: `category_id`, `comment_count`, `likes`, `dislikes`, `ratings_disabled`.

  - Variable objetivo: `views`.



4. **Modelado:**

  - Entrenamiento del modelo utilizando Random Forest Regressor.



5. **Evaluación del Modelo:**

  - Métricas: RMSE, MAE, R².



---



## 5. Modelo de Regresión <a name="modelo"></a>



Se utilizó un modelo de **Random Forest Regressor** para manejar la naturaleza no lineal de los datos y la posible interacción entre variables.



### Código del Modelo:

```python

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import train_test_split



# Selección de características y variable objetivo

X = df[['category_id', 'comment_count', 'likes', 'dislikes', 'ratings_disabled']]

y = df['views']



# División en datos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Entrenamiento del modelo

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)



# Predicción

y_pred = model.predict(X_test)



# Evaluación

rmse = mean_squared_error(y_test, y_pred, squared=False)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

----



```

## 6. Resultados <a name="resultados"></a>

- El modelo predictivo logró un RMSE de **5,509,100**, lo que indica una desviación significativa en las predicciones debido a la alta variabilidad de la variable objetivo (`views`).

- La métrica de R² obtuvo un valor de **[valor calculado]**, lo que sugiere que el modelo explica aproximadamente el **[porcentaje]** de la variabilidad en los datos.

- La inclusión de variables categóricas como `category_id` y numéricas como `comment_count` contribuyó de manera significativa al modelo, pero estas características no capturan toda la complejidad del fenómeno.

- El modelo muestra un mejor desempeño en la predicción de videos con un rango moderado de vistas, mientras que subestima o sobreestima consistentemente los videos con vistas extremadamente altas o bajas.



...



## 7. Conclusiones <a name="conclusiones"></a>

- El análisis ha demostrado que las variables como `comment_count`, `likes` y `dislikes` son importantes para entender el comportamiento de los videos en términos de vistas, pero no explican completamente las diferencias en popularidad entre videos.

- Se observó que videos de ciertas categorías (`category_id`) tienen una mayor propensión a acumular vistas, lo que podría deberse a diferencias en las audiencias de cada tipo de contenido.

- La habilitación o deshabilitación de calificaciones (`ratings_disabled`) parece no tener una relación clara con el número de vistas, aunque podría influir en métricas como "Me gusta" o "Comentarios".

- La distribución de los datos es altamente asimétrica, lo que indica que la mayoría de los videos tienen un número moderado de vistas, mientras que unos pocos acumulan cantidades extremadamente altas. Este fenómeno sugiere la presencia de un efecto de "cola larga".

- El modelo actual tiene limitaciones para capturar la complejidad de los datos debido a la falta de variables explicativas clave, como el tiempo de publicación o palabras clave en títulos/descripciones.



...



## 8. Recomendaciones <a name="recomendaciones"></a>

- **Ampliar las características del modelo:** Incorporar variables adicionales como:

 - Hora y día de publicación del video.

 - Longitud del video (en minutos).

 - Análisis de texto para variables como `title` y `tags`, extrayendo palabras clave relevantes.

- **Transformar los datos:** Aplicar transformaciones logarítmicas a la variable `views` para reducir el impacto de valores extremos y mejorar la estabilidad del modelo.

- **Probar otros modelos predictivos:** Explorar métodos más avanzados como XGBoost o redes neuronales, que pueden capturar relaciones más complejas y no lineales entre variables.

- **Segmentar por categorías:** Entrenar modelos independientes para grupos específicos de videos, como aquellos de categorías altamente populares, para mejorar la precisión dentro de esos segmentos.

- **Realizar análisis temporal:** Evaluar cómo las vistas evolucionan con el tiempo para identificar patrones de crecimiento o saturación.

- **Aumentar la representatividad del conjunto de datos:** Si es posible, añadir más registros para disminuir el sesgo de categorías poco representadas.



...

## 8. Bibliografia <a name="bibliografia"></a>

Brownlee, J. (2021). How to Evaluate Machine Learning Algorithms. Machine Learning Mastery. https://machinelearningmastery.com/evaluate-machine-learning-algorithms/



Aggarwal, C. C. (2015). Data Mining: The Textbook. Springer. (Disponible como referencia en repositorios académicos).



Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288.



Python Software Foundation. (2023). Python Documentation. https://docs.python.org/3/
