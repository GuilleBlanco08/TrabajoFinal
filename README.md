# TrabajoFinal

Trabajo Final – Aprendizaje automático: Random Forest desde cero y redes neuronales para imágenes

Proyecto académico en dos partes que cubre dos enfoques del aprendizaje automático:

Implementación desde cero de un Random Forest en Python, sin usar bibliotecas de machine learning para el algoritmo.
Clasificación de imágenes con redes neuronales, desarrollada en un cuaderno Jupyter.
Parte 1 · Random Forest implementado a mano

Implementación completa del algoritmo Random Forest para entender su funcionamiento interno, en lugar de usarlo como una "caja negra".

Qué incluye:

Árbol de decisión construido de forma recursiva, eligiendo en cada nodo la división que maximiza la ganancia de información según el criterio [Gini / entropía].
Bagging (bootstrap): cada árbol se entrena con una muestra aleatoria con reemplazo del conjunto de entrenamiento.
Selección aleatoria de características en cada división, para que los árboles sean distintos entre sí.
Predicción por votación mayoritaria entre todos los árboles del bosque.
Hiperparámetros configurables: número de árboles, profundidad máxima, mínimo de muestras por nodo y número de características por división.
[Opcional: comparación de resultados con RandomForestClassifier de scikit-learn para validar la implementación.]


Parte 2 · Redes neuronales para clasificación de imágenes

Cuaderno Jupyter (.ipynb) con el diseño, el entrenamiento y la evaluación de redes neuronales para clasificar imágenes.

Qué incluye:

Carga y preprocesamiento de las imágenes (normalización, [redimensionado / aumento de datos]).
Definición de la arquitectura de la red: [red densa / red convolucional (CNN)], con [capas principales].
Entrenamiento con seguimiento de la pérdida y la precisión en entrenamiento y validación.
Evaluación sobre el conjunto de prueba: precisión y matriz de confusión.
[Opcional: comparación entre varias arquitecturas o hiperparámetros.]
 
Framework: [TensorFlow/Keras · PyTorch]
