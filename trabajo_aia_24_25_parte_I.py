#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2024-25
# PARTE I del trabajo práctico: Implementación de árboles de decisión 
#                               y random forests
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: BLANCO DÍAZ
# NOMBRE: GUILLERMO
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS:
# NOMBRE: ENRIQUE
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La
# discusión y el intercambio de información de carácter general con
# los compañeros se permite, pero NO AL NIVEL DE CÓDIGO. Igualmente el
# remitir código de terceros, OBTENIDO A TRAVÉS DE LA RED, o de
# cualquier otro medio, se considerará plagio.

# El objetivo principal del trabajo es reforzar de manera práctica
# los conceptos aprendidos en clase, para alcanzar una mayor
# comprensión de los mismos a través de la implementación que se
# pide. Se permite, si así se desea, el uso de herramientas de
# inteligencia artificial generativa que asistan en el desarrollo
# código, pero esta herramienta ha de usarse sólo como un asistente
# que facilite el trabajo, y en ningún caso se debe entregar un código
# que no se conozca en profundidad y con detalle. 

# Para asegurar que la evaluación del mismo está alineada con el
# objetivo descrito en el párrafo anterior, el trabajo ha de ser
# presentado ante el profesor, explicando con detalle y a nivel de
# código la implementación entregada, y será necesario demostrar total
# comprensión del código entregado. Si el trabajo se hace en grupo,
# ambos miembros del grupo deben poder explicar con detalle de código
# cualquier parte del trabajo.

# Cualquier plagio o entrega de código cuyo funcionamiento no se sea
# capaz de explicar con detalle, significará automáticamente la
# calificación de CERO EN LA ASIGNATURA para TODOS los estudiantes
# involucrados. Independientemente de OTRAS ACCIONES DE CARÁCTER
# DISCIPLINARIO que se pudieran tomar.
# *****************************************************************************************


# MUY IMPORTANTE: 
# ===============    
    
# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. ADEMÁS: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn, salvo donde se dice expresamente.
#   En particular, si se pide implementar algo, se refiere a implementar en python,
#   sin usar Scikit Learn.  
  
# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos 
#   los datasets se suponen dados como arrays de numpy. 

# * Este archivo (con las implementaciones realizadas), ES LO ÚNICO QUE HAY QUE ENTREGAR.

# * AL FINAL DE ESTE ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final a entregar, descomentar esos ejemplos del final y no dejar 
#   ninguna otra ejecución de ejemplos. 



import math
import random
import numpy as np



# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-aia.zip (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN:
    
from carga_datos import *    

# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre pasajeros del Titanic y si sobrevivieron o no. Es una versión 
#   restringida de este conocido dataset, con solo tres caracteristicas:
#   Pclass, IsFemale y Age. Se carga en las variables X_train_titanic, 
#   y_train_titanic, X_test_titanic e y_test_titanic.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos (ver 
#   descripción en votos.py)


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer. 
#   Ver descripcición en sikit learn.

  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    


#  Además, en la carpeta datos/ se tienen los siguientes datasets, que
#  habrán de ser procesado y cargado (es decir, no se caragan directamente con
#  carga_datos.py).   
    
# * Un archivo credito.csv con datos sobre concesión de prestamos en una entidad 
#   bancaria, en función de: tipo de empleo, si ya tiene productos finacieros 
#   contratados, número de propiedades, número de hijos, estado civil y nivel de 
#   ingresos (cargarlo usando pd.read_csv en arrays de numpy X_credito e y_credito,
#   donde X_credito son las seis primeras columnas e y_credito la última).


# * Un archivo adultDataset.csv, con datos de personas para poder predecir si
#   alguien gana más o menos de 50000 dólares anuales, en función de una serie 
#   de características (para más detalles, ver https://archive.ics.uci.edu/dataset/2/adult)  
#   Más adelante se explica cómo cargar y procesar este conjunto de datos. 

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos en archivos de texto. 
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

#  >>>Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#   (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
#  (array([0, 1]), array([136, 229]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([34, 57]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------

def particion_entr_prueba(X, y, test=0.20):
    """
    Separa aleatoriamente (estratificado respecto a y) el conjunto (X, y)
    en dos particiones: entrenamiento y prueba.
    
    Parámetros:
    -----------
    X : np.ndarray de forma (n_ejemplos, n_atributos)
        Matriz de datos.
    y : np.ndarray de forma (n_ejemplos,)
        Vector de etiquetas (pueden ser ints o strings).
    test : float (0 < test < 1)
        Fracción del total que irá a la partición de prueba
        (ejemplo: 0.20 → 20% en prueba, 80% en entrenamiento).
        
    Devuelve:
    ---------
    X_ent, X_pru, y_ent, y_pru : np.ndarray
        Los cuatro arrays resultantes en el orden:
          - X_ent (n_ent × m)
          - X_pru (n_pru × m)
          - y_ent (n_ent,)
          - y_pru (n_pru,)
    """
    # 1) Comprobación de dimensiones
    n_total = X.shape[0]
    if y.shape[0] != n_total:
        raise ValueError("X e y deben tener el mismo número de filas.")

    # 2) Sacar las clases únicas y sus recuentos
    clases_unicas, cuentas_por_clase = np.unique(y, return_counts=True)
    # Ejemplo: clases_unicas = array(['democrata','republicano']),
    #          cuentas_por_clase = array([267, 168])

    # 3) Lists para ir acumulando índices de cada partición
    indices_ent = []  # índices que irán a entrenamiento
    indices_pru = []  # índices que irán a prueba

    # 4) Iterar por cada clase para repartir estratificadamente
    for idx_clase, c in enumerate(clases_unicas):
        # 4.1) Indices de todos los ejemplos de la clase c
        #     np.where(y == c) devuelve una tupla, tomamos [0] para el array.
        indices_clase = np.where(y == c)[0].copy()
        
        # 4.2) Barajar (aleatorizar) esos índices
        np.random.shuffle(indices_clase)

        # 4.3) Calcular cuántos van a la partición de prueba para esta clase
        cuenta_c = cuentas_por_clase[idx_clase]
        n_pru_c = int(np.round(cuenta_c * test))
        # Ejemplo: si cuenta_c = 168 y test = 1/3 ≈ 0.3333 → n_pru_c ≈ 56

        # 4.4) Repartir los índices: primeros n_pru_c a prueba, el resto a entrenamiento
        indices_pru_c = indices_clase[:n_pru_c]
        indices_ent_c = indices_clase[n_pru_c:]

        # 4.5) Agregar a las listas generales
        indices_pru.extend(indices_pru_c.tolist())
        indices_ent.extend(indices_ent_c.tolist())

    # 5) Convertir a arrays y volver a mezclar cada lista por separado
    indices_ent = np.array(indices_ent)
    indices_pru = np.array(indices_pru)

    np.random.shuffle(indices_ent)
    np.random.shuffle(indices_pru)

    # 6) Indexar X e y para obtener los cuatro arrays finales
    X_ent = X[indices_ent]
    y_ent = y[indices_ent]

    X_pru = X[indices_pru]
    y_pru = y[indices_pru]

    return X_ent, X_pru, y_ent, y_pru

# ===============================================
# EJERCICIO 2: IMPLEMENTACIÓN ÁRBOLES DE DECISIÓN
# ===============================================


# En este ejercicio pedimos implementar en python un algoritmo de aprendizaje para árboles 
# de decisión. Los árboles de decisión que trataremos serán árboles binarios, en los que
# en cada nodo interior se pregunta por el valor de un atributo o característica dada, 
# y si ese valor es mayor o menor que un valor umbral dado. Este es el mismo tipo de árbol 
# de decisión que se  manejan en Scikit Learn. 

# Se puede obtener información de este tipo de árboles en la entrada "Decision Trees"
# del manual de Scikit Learn. También en la práctica del Titanic hecha en clase.

# Se propone la implementación de un clasificador basado en árboles de
# de decisión, entrenado usando el algoritmo CART, similar al que implementa 
# la clase DecisonTree de Scikit Learn, pero con ALGUNAS VARIANTES, que indicaremos más
# adelante.

# Los árboles de decisión están formados por nodos. Usar la siguiente clase para la
# implementación de los nodos:
    
class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None,distr=None,*,clase=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izq = izq
        self.der = der
        self.distr= distr
        self.clase = clase
        
    def es_hoja(self):
        return self.clase is not None

# Pasamos a describir los distintos atributos de esta clase:

# - atributo: el atributo por el que se pregunta en el nodo. Referenciaremos a cada
#   atributo POR EL ÍNDICE DE SU POSICIÓN (el número de columna).
# - umbral: es el valor umbral por el que se pregunta en el nodo. Si la instancia tiene un
#   valor de atributo menor o igual que el umbral, se sigue por el subárbol izquierdo. En
#   caso contrario, por el subárbol derecho.
# - izq: es el nodo raiz del subárbol izquierdo.
# - der: el nodo raiz del subárbol derecho.
# - distr: es un diccionario cuyas claves son las posibles clases, y cuyos valores son
#   cuántos ejemplos del conjunto de entrenamiento correspondientes al nodo hay de cada
#   clase. Cuando decimos "ejemplos correspondientes al nodo" queremos decir aquellos que
#   cumplen todas las condiciones (desde la raiz) que llevan a ese nodo.
# - clase: Si el nodo es una hoja, es la clase que predice. Si no es una hoja, este valor es None.



# Lo que sigue es una descripción del algoritmo que se pide implementar para la
# construcción de un árbol de decisión. En principio describiremos la versión básica y más
# conocida, y posteriormente indicaremos las peculiaridades y variantes que pedimos
# introducir a esta versión básica.

# Supondremos que recibimos un conjunto de entrenamiento X,y y además dos valores max_prof
# y min_ejemplos_nodo_interior, que nos van a servir como condiciones adicionales para
# dejar de expandir un nodo. El algoritmo se define recursivamente y tiene además un
# argumento adicional prof (inicialmente 0), con la profundidad del nodo actual.  

# CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0):

# 1. SI prof es mayor o igual que max_prof, 
#       o el número de ejemplos de X es menor que min_ejemplos_nodo_interior,
#       o en X todos los ejemplos son de la misma clase:
#       ENTONCES:
#          Devolver un nodo hoja con la distribución de clases en X,
#                    y con la clase mayoritaria en X
# 2. EN OTRO CASO:
#        encontrar el MEJOR atributo A y el mejor umbral u para ese atributo
#        y particionar en dos tanto X como y:
#            * X_izq, y_izq los ejemplos cuyo valor de A es menor o igual que u
#            * X_der, y_der los ejemplos cuyo valor de A es mayor que u
#        Llamadas recursivas:
#            A_izq=CONSTRUYE_ARBOL(X_izq,y_izq,min_ejemplos_nodo_interior,max_prof,prof+1)
#            A_der=CONSTRUYE_ARBOL(X_der,y_der,min_ejemplos_nodo_interior,max_prof,prof+1)
#        Devolver un nodo interior con el atributo y umbral seleccionado,
#                 con la distribución de clases de X, y con A_izq y A_der
#                 como hijos izquierdo y derecho respectivamente.


# Lo anterior es la descripción básica. A continuación indicamos una serie de variantes y
# cuestiones adicionales que se le piden a esta implementación concreta:

# - Consideraremos la posibilidad de restringir los atributos a usar en el árbol a un
#   número de atributos dado n_atrs. Ese subconjunto de atributos se seleccionará
#   aleatoriamente al principio de la construcción del aŕbol y será el mismo para todos
#   los nodos.
#   Por ejemplo, si el dataset tiene 15 atributos y le damos n_atrs=9, al comienzo de la
#   construcción del árbol seleccionamos aleatoriamente 9 atributos, y ya en los nodos del
#   árbol solo podrán aparecer alguno de esos 9 atributos. Nótese que si n_atrs es igual
#   al total de atributos, tendríamos la versión estándar del algoritmo.
#   NOTA: téngase en cuenta que a diferencia de lo que ocurre en la versión clásica de
#   Random Forests, no sorteamos los atributos en cada nodo, sino que hay un único sorteo
#   inicial para todo el árbol.

# - A la hora de elegir el mejor atributo y umbral para la partición de los nodos
#   interiores, usar el criterio de mejor GANANCIA DE INFORMACIÓN (en particular, NO USAR GINI).

# - La principal carga computacional de este algoritmo se debe a la cantidad de candidatos a
#   mejor atributo y mejor umbral que hay que evaluar en cada nodo, para decidir cuál es
#   la mejor partición. El hecho de limitar el número de atributos candidatos (como se ha
#   descrito más arriba), va en esa dirección. 
#   Otra manera es limitar también los posibles valores umbrales a considerar
#   para cada atributo. Para ello, en la implementación que se pide actuaremos en dos
#   sentidos:
#      (a) Considerar solo como candidatos a umbral los puntos medios entre cada par de 
#         valores consecutivos del atributo en los que hay cambio de clase, para los
#         ejemplos correspondientes a ese nodo.
#         Por ejemplo, si ordenados los valores del atributo A en orden creciente, hay un
#         ejemplo con valor v1 de A y clase C1 y a continuación otro ejemplo con valor v2
#         en A y clase C2 distinta de C1, entonces (v1+v2)/2 es un posible valor umbral
#         candidato. El resto de valores NO se considera candidato.

#      (b) En cada nodo, para elegir los umbrales candidatos correspondientes a un atibuto,
#         no considerar todos los ejemplos que corresponden a ese nodo, sino 
#         sólo  una proporción de los mismos, seleccionada aleatoriamente. La proporción a
#         considerar se da en un parámetro prop_umbral.
#         Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos correspondientes al
#         nodo es de 200 ejemplos, entonces aplicaremos el proceso de selección de
#         umbrales candidatos descrito en (a) considerando sólo un suconjunto de 140
#         ejemplos seleccionado aleatoriamente de entre esos 200.  



# Con las descripciones anteriores, ya podemos precisar lo que se pide en eset apartado. 
# Se pide implementar una clase ArbolDecision con el siguiente formato:
  

# class ArbolDecision:
#     def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
#         ......
#                
#     def entrena(self, X, y):
#         .......
#        
#     def clasifica(self, X):
#         .......
#
#     def clasifica_prob(self, x):
#         .......
#
#     def imprime_arbol(self,nombre_atrs,nombre_clase) :
#         .......



#  El constructor tiene los siguientes argumentos de entrada:

#     + min_ejemplos_nodo_interior: mínimo número de ejemplos del conjunto de 
#       entrenamiento en un nodo del árbol que se aprende, para que se considere 
#       su división.  
#     + max_prof: profundidad máxima del árbol que se aprende.
#     + n_atrs: número de atributos candidatos a considerar en cada partición
#     + prop_umbral: proporción de ejemplos a considerar cuando se buscan los 
#       umbrales candidatos.    
  
#      

# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente.
#     

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Método clasifica_prob: recibe UN EJEMPLO y devuelve un diccionario con la predicción
#   de probabilidad de pertenecer a cada clase. Esa probabilidad se calcula como la
#   proporción de ejemplos de clase en la distribución del nodo hoja que da la
#   predicción.

# * Método imprime_arbol: recibe la lista de nombres de cada atributo (columnas) y el
#   nombre del atributo de clasificación, e imprime el árbol de decisión aprendido 
#   (ver ejemplos más abajo) [SUGERENCIA: hacerlo con una función auxiliar recursiva] 


# Si se llama al método de clasificación, o al de impresión, antes de entrenar el modelo,
# se debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        




# Algunos ejemplos (los resultados pueden variar, debido a la aleatoriedad)
# **************************************************************************

# TITANIC
# -------

# >>> clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
# >>> clf_titanic.entrena(X_train_titanic, y_train_titanic)
# >>> clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Sobrevive")

# Mujer <= 0.000
#      Edad <= 11.000
#           Pclass <= 2.500
#                Sobrevive: 1 -- {1: 10}
#           Pclass > 2.500
#                Sobrevive: 0 -- {0: 13, 1: 8}
#      Edad > 11.000
#           Pclass <= 1.000
#                Sobrevive: 0 -- {0: 62, 1: 30}
#           Pclass > 1.000
#                Sobrevive: 0 -- {0: 270, 1: 32}
# Mujer > 0.000
#      Pclass <= 2.000
#           Edad <= 2.000
#                Sobrevive: 0 -- {0: 1, 1: 1}
#           Edad > 2.000
#                Sobrevive: 1 -- {0: 5, 1: 122}
#      Pclass > 2.000
#           Edad <= 38.500
#                Sobrevive: 1 -- {0: 46, 1: 58}
#           Edad > 38.500
#                Sobrevive: 0 -- {0: 9, 1: 1}

# VOTOS
# -----

# >>> clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
# >>> clf_votos.entrena(Xe_votos, ye_votos)
# >>> nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
# >>> clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")

# Votación 4 <= 0.000
#      Votación 3 <= 0.000
#           Votación 11 <= 0.000
#                Votación 13 <= 0.500
#                     Votación 14 <= -0.500
#                          Partido: democrata -- {'democrata': 2}
#                     Votación 14 > -0.500
#                          Partido: republicano -- {'republicano': 3}
#                Votación 13 > 0.500
#                     Votación 7 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 7 > -1.000
#                          Partido: democrata -- {'democrata': 4}
#           Votación 11 > 0.000
#                Partido: democrata -- {'democrata': 11}
#      Votación 3 > 0.000
#           Partido: democrata -- {'democrata': 149}
# Votación 4 > 0.000
#      Votación 11 <= 0.500
#           Votación 10 <= -1.000
#                Votación 12 <= -1.000
#                     Votación 3 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 3 > -1.000
#                          Partido: republicano -- {'republicano': 2}
#                Votación 12 > -1.000
#                     Votación 3 <= 0.000
#                          Partido: republicano -- {'republicano': 35}
#                     Votación 3 > 0.000
#                          Partido: republicano -- {'democrata': 1, 'republicano': 2}
#           Votación 10 > -1.000
#                Partido: republicano -- {'republicano': 55}
#      Votación 11 > 0.500
#           Votación 7 <= -1.000
#                Votación 3 <= -1.000
#                     Votación 13 <= 0.000
#                          Partido: democrata -- {'democrata': 1}
#                     Votación 13 > 0.000
#                          Partido: republicano -- {'democrata': 2, 'republicano': 9}
#                Votación 3 > -1.000
#                     Partido: democrata -- {'democrata': 6}
#           Votación 7 > -1.000
#                Partido: republicano -- {'republicano': 4}


# IRIS
# ----

    
# >>> clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# >>> clf_iris.entrena(X_train_iris, y_train_iris)
# >>> clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")



#  Long. Pétalo <= 2.450
#       Clase: 0 -- {0: 33}
#  Long. Pétalo > 2.450
#       Long. Pétalo <= 4.900
#            Anch. Pétalo <= 1.650
#                 Clase: 1 -- {1: 32}
#            Anch. Pétalo > 1.650
#                 Clase: 2 -- {1: 1, 2: 3}
#       Long. Pétalo > 4.900
#            Clase: 2 -- {2: 30}


# CÁNCER DE MAMA
# --------------

# >>> clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer.entrena(Xev_cancer, yev_cancer)

# >>> nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension',
#        'radius error', 'texture error', 'perimeter error', 'area error',
#        'smoothness error', 'compactness error', 'concavity error',
#        'concave points error', 'symmetry error',
#        'fractal dimension error', 'worst radius', 'worst texture',
#        'worst perimeter', 'worst area', 'worst smoothness',
#        'worst compactness', 'worst concavity', 'worst concave points',
#        'worst symmetry', 'worst fractal dimension']

# >>> clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")


#  mean concave points <= 0.051
#       mean area <= 696.050
#            area error <= 34.405
#                 mean area <= 505.550
#                      Es benigno: 1 -- {1: 172}
#                 mean area > 505.550
#                      worst texture <= 30.145
#                           mean concave points <= 0.050
#                                Es benigno: 1 -- {1: 63}
#                           mean concave points > 0.050
#                                Es benigno: 0 -- {0: 1, 1: 1}
#                      worst texture > 30.145
#                           mean texture <= 24.840
#                                compactness error <= 0.013
#                                     Es benigno: 0 -- {0: 3}
#                                compactness error > 0.013
#                                     Es benigno: 1 -- {1: 2}
#                           mean texture > 24.840
#                                Es benigno: 1 -- {1: 11}
#            area error > 34.405
#                 mean concave points <= 0.032
#                      Es benigno: 1 -- {1: 7}
#                 mean concave points > 0.032
#                      mean perimeter <= 89.175
#                           Es benigno: 0 -- {0: 3}
#                      mean perimeter > 89.175
#                           mean texture <= 20.115
#                                Es benigno: 1 -- {1: 3}
#                           mean texture > 20.115
#                                Es benigno: 0 -- {0: 1}
#       mean area > 696.050
#            mean texture <= 16.190
#                 Es benigno: 1 -- {1: 4}
#            mean texture > 16.190
#                 worst fractal dimension <= 0.066
#                      Es benigno: 1 -- {1: 2}
#                 worst fractal dimension > 0.066
#                      Es benigno: 0 -- {0: 6}
#  mean concave points > 0.051
#       mean area <= 790.850
#            worst texture <= 25.655
#                 mean concave points <= 0.079
#                      mean concave points <= 0.052
#                           Es benigno: 0 -- {0: 1}
#                      mean concave points > 0.052
#                           Es benigno: 1 -- {1: 20}
#                 mean concave points > 0.079
#                      Es benigno: 0 -- {0: 6}
#            worst texture > 25.655
#                 perimeter error <= 1.558
#                      Es benigno: 0 -- {0: 1, 1: 1}
#                 perimeter error > 1.558
#                      Es benigno: 0 -- {0: 37}
#       mean area > 790.850
#            Es benigno: 0 -- {0: 111}



# EJEMPLOS DE RENDIMIENTOS OBTENIDOS CON LOS CLASIFICADORES:
# ----------------------------------------------------------

# Usamos la siguiente función para medir el rendimiento (proporción de aciertos) 
# de un clasificador sobre un conjunto de ejemplos:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/X.shape[0]
    

# Ejemplos (obviamente, el resultado puede variar):


# >>> rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
# 0.8158682634730539
# >>> rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
# 0.7982062780269058

# >>> rendimiento(clf_votos,Xe_votos,ye_votos)
# 0.9827586206896551
# >>> rendimiento(clf_votos,Xp_votos,yp_votos)
# 0.9310344827586207

# >>> rendimiento(clf_iris,X_train_iris,y_train_iris)
#  0.98989898989899
# >>> rendimiento(clf_iris,X_test_iris,y_test_iris)
# 0.9607843137254902

# >>> rendimiento(clf_cancer,Xev_cancer,yev_cancer)
# 0.9956140350877193
# >>> rendimiento(clf_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806


# Función auxiliar para calcular entropía en base 2:
def entropia(y_sub):
    """
    Calcula la entropía de un vector de etiquetas y_sub (1D), usando base 2:
        H = -∑ p_c · log2(p_c),
    donde p_c = (#ejemplos de clase c) / (total de ejemplos en y_sub).
    Ignora probabilidades p_c == 0 para evitar log2(0).
    """
    clases_sub, cuentas_sub = np.unique(y_sub, return_counts=True)
    if cuentas_sub.size == 0:
        return 0.0
    p = cuentas_sub.astype(float) / cuentas_sub.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class ArbolDecision:
    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
        """
        Constructor del árbol de decisión.

        Parámetros:
        -----------
        min_ejemplos_nodo_interior : int
            Número mínimo de ejemplos que debe tener un nodo para que se intente dividirlo.
            Si en un nodo llegan menos de este umbral, se convierte en hoja inmediatamente.
        max_prof : int
            Profundidad máxima permitida. Si la llamada recursiva alcanza prof ≥ max_prof,
            el nodo se convierte en hoja.
        n_atrs : int
            Número de atributos (columnas) que se seleccionarán aleatoriamente al principio del
            entrenamiento y que se usarán en TODO el árbol. Si n_atrs ≥ número de columnas
            de X, se usan todas las columnas.
        prop_umbral : float
            Proporción (0 < prop_umbral ≤ 1) de ejemplos de cada nodo que se usarán para
            hallar candidatos de umbral. Ejemplo: prop_umbral=0.7 en un nodo con 200 ejemplos
            usa 140 aleatorios para generar “cortes” candidatos de ese atributo.
        """
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof                  = max_prof
        self.n_atrs                    = n_atrs
        self.prop_umbral               = prop_umbral

        # Hasta que se entrene, self.raiz queda en None:
        self.raiz = None

        # Aquí guardaremos el array de índices de atributos candidatos (subconjunto aleatorio).
        self.atributos_candidatos = None

    def entrena(self, X, y):
        """
        Entrena el árbol de decisión con los datos (X, y).

        Pasos:
        1) Obtener m = número total de columnas (atributos) en X.
        2) Si n_atrs < m, sortear n_atrs índices de columna (0..m-1) sin reemplazo
           y guardarlos en self.atributos_candidatos. Si n_atrs ≥ m, usar todos los atributos.
        3) Llamar recursivamente a _construye_arbol_rec(X, y, prof=0) y guardar la raíz en self.raiz.
        """
        n_ejemplos, n_atributos_totales = X.shape

        if self.n_atrs < n_atributos_totales:
            self.atributos_candidatos = np.random.choice(
                a       = n_atributos_totales,
                size    = self.n_atrs,
                replace = False
            )
        else:
            self.atributos_candidatos = np.arange(n_atributos_totales)

        # Construcción recursiva de todo el árbol:
        self.raiz = self._construye_arbol_rec(X, y, prof=0)

    def _construye_arbol_rec(self, X_n, y_n, prof):
        """
        Construye recursivamente un nodo (hoja o interior) del árbol.
        Parámetros:
          - X_n : array numpy de forma (n_nodo, m), subconjunto de ejemplos en el nodo.
          - y_n : array numpy de forma (n_nodo,), etiquetas correspondientes a X_n.
          - prof: int, profundidad actual del nodo (0 = raíz).
        Retorna:
          - Un objeto Nodo, que puede ser hoja o nodo interior con atributo/umbral.
        """

        # 1) Calculamos distribución de clases en este nodo (para hoja o para calculo de distr).
        clases_sub, cuentas_sub = np.unique(y_n, return_counts=True)
        dict_distr_padre = {clase: int(cuenta) for clase, cuenta in zip(clases_sub, cuentas_sub)}

        # 2) Condiciones de parada (creación de hoja):
        #    a) prof ≥ max_prof
        #    b) len(y_n) < min_ejemplos_nodo_interior
        #    c) todas las etiquetas de y_n son iguales (pureza total)
        if (prof >= self.max_prof) or (len(y_n) < self.min_ejemplos_nodo_interior) or (clases_sub.size == 1):
            idx_mayor = np.argmax(cuentas_sub)
            clase_mayoritaria = clases_sub[idx_mayor]
            return Nodo(
                atributo = None,
                umbral   = None,
                izq      = None,
                der      = None,
                distr    = dict_distr_padre,
                clase    = clase_mayoritaria
            )

        # 3) Si no paramos, buscamos el MEJOR atributo A y umbral u que maximicen la ganancia de información
        mejor_gain   = -np.inf
        mejor_A      = None
        mejor_umbral = None
        H_padre      = entropia(y_n)

        # 4) Para cada atributo candidato (subconjunto sorteado en entrena()):
        for A in self.atributos_candidatos:
            n_nodo = X_n.shape[0]
            # 4.1) Submuestreo de ejemplos para hallar candidatos de umbral:
            k = max(1, int(np.round(self.prop_umbral * n_nodo)))
            indices_muestra = np.random.choice(n_nodo, size=k, replace=False)

            Xm = X_n[indices_muestra, A]
            ym = y_n[indices_muestra]

            # 4.2) Ordenar la muestra por valor de Xm para detectar cambios de clase vecinos
            orden_muestra = np.argsort(Xm)
            Xm_orden = Xm[orden_muestra]
            ym_orden = ym[orden_muestra]

            # 4.3) Construir lista de umbrales candidatos: puntos medios donde cambia la clase
            umbrales_A = []
            for i in range(len(Xm_orden) - 1):
                if ym_orden[i] != ym_orden[i + 1]:
                    u = (Xm_orden[i] + Xm_orden[i + 1]) / 2.0
                    umbrales_A.append(u)

            if len(umbrales_A) == 0:
                continue  # No hay cambio de clase en la muestra → ningún candidato para este atributo

            # 4.4) Para cada candidato u, calculamos la ganancia de información
            for u in umbrales_A:
                mask_izq = (X_n[:, A] <= u)
                mask_der = ~mask_izq
                if (np.sum(mask_izq) == 0) or (np.sum(mask_der) == 0):
                    continue  # Split inválido (alguna rama vacía)

                y_izq = y_n[mask_izq]
                y_der = y_n[mask_der]

                H_izq = entropia(y_izq)
                H_der = entropia(y_der)

                N_nodo = len(y_n)
                N_izq  = len(y_izq)
                N_der  = len(y_der)

                H_hijos = (N_izq / N_nodo) * H_izq + (N_der / N_nodo) * H_der
                gain    = H_padre - H_hijos

                if gain > mejor_gain:
                    mejor_gain   = gain
                    mejor_A      = int(A)
                    mejor_umbral = float(u)

        # 5) Si no encontramos ningún split con ganancia positiva, devolvemos un nodo hoja
        if (mejor_A is None) or (mejor_gain <= 0):
            idx_mayor = np.argmax(cuentas_sub)
            clase_mayoritaria = clases_sub[idx_mayor]
            return Nodo(
                atributo = None,
                umbral   = None,
                izq      = None,
                der      = None,
                distr    = dict_distr_padre,
                clase    = clase_mayoritaria
            )

        # 6) Si sí hay mejor_A y mejor_umbral, particionamos todo el conjunto (X_n, y_n)
        mask_split = (X_n[:, mejor_A] <= mejor_umbral)
        X_izq, y_izq = X_n[mask_split],    y_n[mask_split]
        X_der, y_der = X_n[~mask_split],   y_n[~mask_split]

        # Llamadas recursivas a la izquierda y a la derecha (prof + 1)
        nodo_izq = self._construye_arbol_rec(X_izq, y_izq, prof + 1)
        nodo_der = self._construye_arbol_rec(X_der, y_der, prof + 1)

        # Devolvemos un nodo interior con el atributo y umbral seleccionados
        return Nodo(
            atributo = mejor_A,
            umbral   = mejor_umbral,
            izq      = nodo_izq,
            der      = nodo_der,
            distr    = dict_distr_padre,
            clase    = None
        )

    def clasifica(self, X):
        """
        Clasifica un array X (N × m) y devuelve un array de longitud N con las predicciones.
        Si self.raiz es None (no se ha entrenado), lanza ClasificadorNoEntrenado.
        """
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")

        n_ejemplos = X.shape[0]
        preds = np.empty(shape=(n_ejemplos,), dtype=object)

        for i in range(n_ejemplos):
            nodo = self.raiz
            # Descender por el árbol hasta llegar a una hoja
            while not nodo.es_hoja():
                if X[i, nodo.atributo] <= nodo.umbral:
                    nodo = nodo.izq
                else:
                    nodo = nodo.der
            preds[i] = nodo.clase

        return preds

    def clasifica_prob(self, x):
        """
        Clasifica un único ejemplo x (vector 1D de longitud m).
        Devuelve un dict {clase_i: probabilidad_i}, usando la distribución en la hoja.
        Si self.raiz es None, lanza ClasificadorNoEntrenado.
        """
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")

        nodo = self.raiz
        while not nodo.es_hoja():
            if x[nodo.atributo] <= nodo.umbral:
                nodo = nodo.izq
            else:
                nodo = nodo.der

        dist = nodo.distr
        total = sum(dist.values())
        return {clase: cuenta / total for clase, cuenta in dist.items()}

    def imprime_arbol(self, nombre_atrs, nombre_clase):
        """
        Imprime en consola un dibujo textual del árbol entrenado.
        Parámetros:
          - nombre_atrs : lista de strings con el nombre de cada columna de X.
          - nombre_clase: string con el nombre de la variable objetivo.
        Si self.raiz es None, lanza ClasificadorNoEntrenado.
        """
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")
        self._imprime_nodo(self.raiz, prof=0, nombre_atrs=nombre_atrs, nombre_clase=nombre_clase)

    def _imprime_nodo(self, nodo, prof, nombre_atrs, nombre_clase):
        """
        Función auxiliar recursiva que imprime cada nodo con indentación.
        - Si nodo.es_hoja(): imprime 
            "<espacios><nombre_clase>: <etiqueta_mayoritaria> -- {distr}"
        - Si no, imprime 
            "<espacios><atributo> <= <umbral>"
            (hijo izquierdo)
            "<espacios><atributo> > <umbral>"
            (hijo derecho)
        donde espacios = "    " * prof.
        """
        espacios = "    " * prof

        if nodo.es_hoja():
            print(f"{espacios}{nombre_clase}: {nodo.clase} -- {nodo.distr}")
        else:
            atr = nombre_atrs[nodo.atributo]
            um  = nodo.umbral
            print(f"{espacios}{atr} <= {um:.3f}")
            self._imprime_nodo(nodo.izq, prof + 1, nombre_atrs, nombre_clase)
            print(f"{espacios}{atr} > {um:.3f}")
            self._imprime_nodo(nodo.der, prof + 1, nombre_atrs, nombre_clase)









# =============================================
# EJERCICIO 3: IMPLEMENTACIÓN DE RANDOM FORESTS
# =============================================

# Usando la clase ArbolDecision, implementar un clasificador Random Forest. 

# Un clasificador Random Forest aplica dos técnicas que reducen el sobreajuste que 
# pudiéramos tener con un único árbol de decisión:

# - En lugar de aprender un árbol. se aprenden varios árboles y a la hora de clasificar
#   nuevos ejemplos, se devuelve la clasificación mayoritaria.
# - Cada uno de esos árboles no se aprende con el conjunto de entrenamiento original, sino
#   con una muestra de ejemplos, obtenido seleccionado los ejemplos aleatoriamente del 
#   conjunto total, CON REEMPLAZO. Además, durante el aprendizaje y en cada nodo, no se usan todos
#   los atributos sino un sunconjunto de ellos obtenidos aleatoriamente (el mismo para todo el árbol). 

# NOTA IMPORTANTE: En la versión estándar del algoritmo Random Forest, el subconjunto de
# atributos a considerar se sortea EN CADA NODO de los árboles que se aprenden. Sin
# embargo, en nuestro caso, como vamos a usar la clase ArbolDecision del ejercicio
# anterior, se va usar el mismo subconjunto de atributos EN CADA ÁRBOL APRENDIDO.

# Concretando, se pide implementar una clase RandomForest con la siguiente estructura:


# class RandomForest:
#     def __init__(self, n_arboles=5,prop_muestras=1.0,
#                        min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
#         .......                   

#     def entrena(self, X, y):
#         .......

#     def clasifica(self, X):
#         .......
    
# Los argumentos del constructor son:

# - n_arboles: el número de árboles que se van a obtener para el clasificador.
# - n_muestras: el número de ejemplos a muestrear para el aprendizaje de cada árbol.
# - El resto de argumentos son los mismos que en el ejercicio anterior, y se usan en el
#   aprendizaje de cada árbol.


# Ejemplos:
# *********

# VOTOS:
# ------

# >>> clf_votos_rf=RandomForest(n_arboles=10,min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=6,prop_umbral=0.8)
# >>> clf_votos_rf.entrena(Xe_votos, ye_votos)
# >>> rendimiento(clf_votos_rf,Xe_votos,ye_votos)
# 0.9517241379310345
# >>> rendimiento(clf_votos_rf,Xp_votos,yp_votos)
# 0.9586206896551724


# >>> clf_cancer_rf = RandomForest(n_arboles=15,min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer_rf.entrena(Xev_cancer, yev_cancer)
# >>> rendimiento(clf_cancer_rf,Xev_cancer,yev_cancer)
# 1.0
# >>> rendimiento(clf_cancer_rf,Xp_cancer,yp_cancer)
# 0.9911504424778761


#------------------------------------------------------------------------------

class RandomForest:
    def __init__(self, n_arboles=5, prop_muestras=1.0,
                 min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
        """
        Constructor del clasificador Random Forest.

        Parámetros:
        -----------
        n_arboles : int
            Número de árboles en el bosque.
        prop_muestras : float (0 < prop_muestras ≤ 1)
            Proporción de ejemplos del conjunto de entrenamiento original que se
            seleccionarán (con reemplazo) para entrenar cada árbol (bootstrap).
        min_ejemplos_nodo_interior : int
            Se pasa a cada ArbolDecision: número mínimo de ejemplos para dividir un nodo.
        max_prof : int
            Se pasa a cada ArbolDecision: profundidad máxima del árbol.
        n_atrs : int
            Se pasa a cada ArbolDecision: número de atributos candidatos por árbol.
        prop_umbral : float (0 < prop_umbral ≤ 1)
            Se pasa a cada ArbolDecision: proporción de ejemplos a usar en cada nodo
            para generar candidatos de umbral.
        """
        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral

        # Lista donde guardaremos cada árbol entrenado
        self.arboles = []

    def entrena(self, X, y):
        """
        Entrena el Random Forest sobre los datos (X, y). Por cada uno de los n_arboles:
          1) Crear un bootstrap sample de X, y con reemplazo, de tamaño n_bootstrap = round(prop_muestras * n_total).
          2) Instanciar un ArbolDecision con los hiperparámetros dados.
          3) Entrenar ese árbol sobre el bootstrap sample.
          4) Guardar el árbol entrenado en self.arboles.
        """
        n_total = X.shape[0]
        # Número de ejemplos en cada bootstrap
        n_bootstrap = max(1, int(round(self.prop_muestras * n_total)))

        self.arboles = []
        for _ in range(self.n_arboles):
            # 1) Muestreo con reemplazo: índices aleatorios de tamaño n_bootstrap
            idxs = np.random.choice(n_total, size=n_bootstrap, replace=True)
            X_bs = X[idxs]
            y_bs = y[idxs]

            # 2) Crear el árbol con los mismos hiperparámetros
            árbol = ArbolDecision(
                min_ejemplos_nodo_interior=self.min_ejemplos_nodo_interior,
                max_prof=self.max_prof,
                n_atrs=self.n_atrs,
                prop_umbral=self.prop_umbral
            )
            # 3) Entrenar sobre el bootstrap sample
            árbol.entrena(X_bs, y_bs)

            # 4) Guardar el árbol entrenado
            self.arboles.append(árbol)

    def clasifica(self, X):
        """
        Clasifica un array X (N × m) devolviendo para cada instancia la clase
        por mayoría de votos entre los árboles del bosque.

        Si no se ha llamado a entrena() con anterioridad (self.arboles vacío),
        lanza ClasificadorNoEntrenado.
        """
        if len(self.arboles) == 0:
            raise ClasificadorNoEntrenado("El Random Forest no ha sido entrenado aún.")

        n_ejemplos = X.shape[0]
        n_trees = len(self.arboles)

        # 1) Cada árbol predice sobre todo X
        #    Guardamos un listado de arrays, cada uno con shape (n_ejemplos,)
        preds_por_arbol = [árbol.clasifica(X) for árbol in self.arboles]

        # 2) Para cada instancia i, hacemos majority vote entre preds_por_arbol[*][i]
        preds_final = np.empty(shape=(n_ejemplos,), dtype=object)
        for i in range(n_ejemplos):
            votos = [preds_por_arbol[t][i] for t in range(n_trees)]
            # Contar manualmente sin usar Counter
            conteo = {}
            for voto in votos:
                if voto in conteo:
                    conteo[voto] += 1
                else:
                    conteo[voto] = 1

            # Obtener la clase con mayor frecuencia
            clase_pred = None
            max_cuenta = -1
            for clase, cuenta in conteo.items():
                if cuenta > max_cuenta:
                    max_cuenta = cuenta
                    clase_pred = clase

            preds_final[i] = clase_pred

        return preds_final



# =========================================
# EJERCICIO 4: AJUSTANDO LOS CLASIFICADORES
# =========================================

# En este ejercicio vamos a tratar de obtener buenos clasificadores para los 
# los siguientes conjuntos de datos: IMDB, credito, AdultDataset y dígitos.

# ---------------------------
# 4.1 PREPARANDO LOS DATASETS     
# ---------------------------

# Excepto a IMDB, que ya se carga cuando se ejecuta carga_datos.py, el resto 
# tendremos que hacer antes algún preprocesado:
    
# - En X_credito, los atributos son categóricos, así que hay que transformarlos 
#   en numéricos para que se puedan usar con nuestros árboles de decisión. 
#   En el caso de árboles de decisión no es necesario hacer "one hot encoding",
#   sino que basta con codificar los valores de los atributos con números naturales
#   Para ello, SE PIDE USAR el OrdinalEncoder de sklearn.preprocessing (ver manual). 
#   Será necesario también separar en conjunto de prueba y de entrenamiento y
#   validación. 

# - El dataset AdultDataset nos viene es un archivo csv. Cargarlo con 
#   read_csv de pandas, separarlo en entrenamiento y prueba 
#   y aplicarle igualmente OrdinalEncoder, pero sólo a las características desde la 
#   quinta en adelante (ya que las cuatro primeras columnas ya son numéricas). 

# - El dataset de dígitos los podemos obtener a partir de los datos que están en 
#   la carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#   píxeles, y cada pixel vendrá representado por un caracter "espacio en
#   blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#   (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#   (es decir, no distinguiremos entre el borde y el interior). En cada
#   conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#   clasificaciones de cada imagen (es decir, el número que representan) vienen
#   en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#   funciones python que lean esos ficheros y obtengan los datos en el mismo
#   formato numpy en el que los necesita el clasificador. 
#   Los datos están ya separados en entrenamiento, validación y prueba. 

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

#   Se pide incluir aquí las definiciones y órdenes necesarias para definir
#   las siguientes variables, con los datasets anteriores como arrays de numpy.


# * X_train_credito, y_train_credito, X_test_credito, y_test_credito
#   conteniendo el dataset de crédito con los atributos numñericos:

# --- 1) DATASET CRÉDITO -------------------------------------------------------

# Cargar el CSV: asumimos que está en "datos/credito.csv"
df_credito = pd.read_csv("datos/credito.py", header=None)

# Separar X_credito (seis primeras columnas) y y_credito (última columna)
X_credito = df_credito.iloc[:, :-1].to_numpy()
y_credito = df_credito.iloc[:, -1].to_numpy()

# Codificar las 6 columnas categóricas con valores ordinales
ordinal_cred = OrdinalEncoder()
X_credito_num = ordinal_cred.fit_transform(X_credito)

# 1.1) Partir en entrenamiento+validación (80%) y prueba (20%)
X_temp_credito, X_test_credito, y_temp_credito, y_test_credito = particion_entr_prueba(
    X_credito_num, y_credito, test=0.20
)

# 1.2) De los 80% (X_temp_credito), partir 75% para entrenamiento y 25% para validación
#      (25% de 80% = 20% del total). Es decir, usamos test=0.25 sobre el subset.
X_train_credito, X_valid_credito, y_train_credito, y_valid_credito = particion_entr_prueba(
    X_temp_credito, y_temp_credito, test=0.25
)


# * X_train_adult, y_train_adult, X_test_adult, y_test_adult
#   conteniendo el AdultDataset con los atributos numéricos:

# --- 2) DATASET ADULTDATASET --------------------------------------------------

# Cargar el CSV: asumimos que está en "datos/adultDataset.csv"
df_adult = pd.read_csv("datos/adultDataset.csv", header=None)

# Separar X_adult (todas menos la última) e y_adult (última columna)
X_adult = df_adult.iloc[:, :-1].to_numpy()
y_adult = df_adult.iloc[:, -1].to_numpy()

# Partir en entrenamiento (70%) y prueba (30%)
X_train_adult, X_test_adult, y_train_adult, y_test_adult = particion_entr_prueba(
    X_adult, y_adult, test=0.30
)

# Aplicar OrdinalEncoder solo a las columnas categóricas (índices 4 en adelante)
ordinal_adult = OrdinalEncoder()
ordinal_adult.fit(X_train_adult[:, 4:])
X_train_adult[:, 4:] = ordinal_adult.transform(X_train_adult[:, 4:])
X_test_adult[:, 4:]  = ordinal_adult.transform(X_test_adult[:, 4:])



# * X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
#   conteniendo el dataset de los dígitos escritos a mano:
    

# --- 3) DATASET DÍGITOS (digitdata) ------------------------------------------

# Función auxiliar para leer un fichero de imágenes de dígitos en formato texto:
def lee_imagenes_digitos(path_imagenes, n_ejemplos):
    """
    Lee n_ejemplos imágenes de 28×28 caracteres del fichero path_imagenes.
    Cada imagen ocupa 28 líneas de 28 caracteres. Un píxel blanco (' ') → 0; '+' o '#' → 1.
    Devuelve un array numpy de forma (n_ejemplos, 784).
    """
    X = np.zeros((n_ejemplos, 28 * 28), dtype=int)
    with open(path_imagenes, 'r') as f:
        linea = f.readline()
        i = 0
        while linea and i < n_ejemplos:
            pixeles = []
            for _ in range(28):
                fila = linea.rstrip("\n")
                pixeles.extend([0 if ch == ' ' else 1 for ch in fila])
                linea = f.readline()
            X[i, :] = np.array(pixeles, dtype=int)
            i += 1
            linea = f.readline()
    return X

# Función auxiliar para leer las etiquetas de dígitos:
def lee_labels_digitos(path_labels):
    """
    Lee un fichero de etiquetas (una línea = un dígito). Devuelve un array numpy.
    """
    with open(path_labels, 'r') as f:
        y = [int(l.strip()) for l in f.readlines()]
    return np.array(y, dtype=int)

# Rutas a los archivos digitdata:
#   "datos/digitdata/trainimages"  → imágenes de entrenamiento
#   "datos/digitdata/trainlabels"  → etiquetas entrenamiento
#   "datos/digitdata/validimages"  → imágenes de validación
#   "datos/digitdata/validlabels"  → etiquetas validación
#   "datos/digitdata/testimages"   → imágenes de prueba
#   "datos/digitdata/testlabels"   → etiquetas prueba

# 3.1) Leer etiquetas para contar cuántas hay en cada fichero
y_train_digitos = lee_labels_digitos("datos/digitdata/trainlabels")
y_valid_digitos = lee_labels_digitos("datos/digitdata/validlabels")
y_test_digitos  = lee_labels_digitos("datos/digitdata/testlabels")

n_train_d = y_train_digitos.shape[0]
n_valid_d = y_valid_digitos.shape[0]
n_test_d  = y_test_digitos.shape[0]

# 3.2) Leer las imágenes correspondientes
X_train_digitos = lee_imagenes_digitos("datos/digitdata/trainimages", n_train_d)
X_valid_digitos = lee_imagenes_digitos("datos/digitdata/validimages", n_valid_d)
X_test_digitos  = lee_imagenes_digitos("datos/digitdata/testimages",  n_test_d)

# Renombrar variables según pide el enunciado:
X_train_dg, y_train_dg = X_train_digitos, y_train_digitos
X_valid_dg, y_valid_dg = X_valid_digitos, y_valid_digitos
X_test_dg,  y_test_dg  = X_test_digitos,  y_test_digitos






# -----------------------------
# 4.2 AJUSTE DE HIPERPARÁMETROS     
# -----------------------------

# En nuestra implementación de RandomForest tenemos los siguientes 
# hiperparámetros: 

# n_arboles
# prop_muestras
# min_ejemplos_nodo_interior
# max_prof
# n_atrs
# prop_umbral

# Se trata ahora de encontrar, en cada dataset, una buena combinación de valores para esos 
# hiperparámetros, tratando de obtener un buen rendimiento de los clasificadores. Hacerlo
# usando un conjunto de validación: según se ha visto en la teoría, esto consiste en particionar  
# en entrenamiento, validación y prueba, entrenando por cada combinación de hiperparámetros 
# con el conjunto de entrenamiento y evaluando el rendimiento en validación. El entrenamiento final 
# con la mejor combinación ha de hacerse en la unión de entrenamiento y validación.
    

# NO ES NECESARIO ser demasiado exhaustivo, basta con probar algunas combinaciones, 
# pero sí es importante describir el proceso realizado y las mejores combinaciones 
# encontradas en cada caso. 
# DEJAR ESTE APARTADO COMENTADO, para que no se ejecuten las pruebas realizadas cuando se cargue
# el archivo. 

# ----------------------------



# -- CRÉDITO --
# Partición entrenamiento+validación / prueba
Xe_cred, Xp_cred, ye_cred, yp_cred = particion_entr_prueba(X_credito, y_credito, test=0.2)
# A su vez, Xe_cred se divide en entrenamiento / validación
Xe_cred_ent, Xe_cred_val, ye_cred_ent, ye_cred_val = particion_entr_prueba(Xe_cred, ye_cred, test=0.25)

# Listas de valores a probar
lista_n_arboles = [10, 20, 50]
lista_prop_muestras = [0.6, 0.8, 1.0]
lista_min_ej = [3, 5, 10]
lista_max_prof = [5, 10]
lista_n_atrs = [5, 10, X_credito.shape[1]]
lista_prop_umb = [0.6, 0.8, 1.0]

mejor_comb = None
mejor_rend_val = 0.0

for n_ar in lista_n_arboles:
    for pm in lista_prop_muestras:
        for min_e in lista_min_ej:
            for mp in lista_max_prof:
                for na in lista_n_atrs:
                    for pu in lista_prop_umb:
                        rf = RandomForest(
                            n_arboles=n_ar,
                            prop_muestras=pm,
                            min_ejemplos_nodo_interior=min_e,
                            max_prof=mp,
                            n_atrs=na,
                            prop_umbral=pu
                        )
                        rf.entrena(Xe_cred_ent, ye_cred_ent)
                        rend_val = rendimiento(rf, Xe_cred_val, ye_cred_val)
                        if rend_val > mejor_rend_val:
                            mejor_rend_val = rend_val
                            mejor_comb = (n_ar, pm, min_e, mp, na, pu)

print("Mejor combinación en 'crédito':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

# Reentreno definitivo sobre entrenamiento+validación
(n_ar, pm, min_e, mp, na, pu) = mejor_comb
RF_CREDITO = RandomForest(
    n_arboles=n_ar,
    prop_muestras=pm,
    min_ejemplos_nodo_interior=min_e,
    max_prof=mp,
    n_atrs=na,
    prop_umbral=pu
)
Xcred_ent_val = np.vstack((Xe_cred_ent, Xe_cred_val))
ycred_ent_val = np.hstack((ye_cred_ent, ye_cred_val))
RF_CREDITO.entrena(Xcred_ent_val, ycred_ent_val)
print("Rendimiento RF sobre crédito (entren+valid → prueba):", rendimiento(RF_CREDITO, Xp_cred, yp_cred))
print("\n")

# -- ADULTDATASET --
# Partición entrenamiento+validación / prueba
Xe_adult, Xp_adult, ye_adult, yp_adult = particion_entr_prueba(X_train_adult, y_train_adult, test=0.2)
Xe_adult_ent, Xe_adult_val, ye_adult_ent, ye_adult_val = particion_entr_prueba(Xe_adult, ye_adult, test=0.25)

lista_n_arboles = [10, 20, 50]
lista_prop_muestras = [0.6, 0.8, 1.0]
lista_min_ej = [5, 10]
lista_max_prof = [5, 10]
lista_n_atrs = [5, 10, X_train_adult.shape[1]]
lista_prop_umb = [0.6, 0.8, 1.0]

mejor_comb = None
mejor_rend_val = 0.0

for n_ar in lista_n_arboles:
    for pm in lista_prop_muestras:
        for min_e in lista_min_ej:
            for mp in lista_max_prof:
                for na in lista_n_atrs:
                    for pu in lista_prop_umb:
                        rf = RandomForest(
                            n_arboles=n_ar,
                            prop_muestras=pm,
                            min_ejemplos_nodo_interior=min_e,
                            max_prof=mp,
                            n_atrs=na,
                            prop_umbral=pu
                        )
                        rf.entrena(Xe_adult_ent, ye_adult_ent)
                        rend_val = rendimiento(rf, Xe_adult_val, ye_adult_val)
                        if rend_val > mejor_rend_val:
                            mejor_rend_val = rend_val
                            mejor_comb = (n_ar, pm, min_e, mp, na, pu)

print("Mejor combinación en 'adult':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

(n_ar, pm, min_e, mp, na, pu) = mejor_comb
RF_ADULT = RandomForest(
    n_arboles=n_ar,
    prop_muestras=pm,
    min_ejemplos_nodo_interior=min_e,
    max_prof=mp,
    n_atrs=na,
    prop_umbral=pu
)
Xadult_ent_val = np.vstack((Xe_adult_ent, Xe_adult_val))
yadult_ent_val = np.hstack((ye_adult_ent, ye_adult_val))
RF_ADULT.entrena(Xadult_ent_val, yadult_ent_val)
print("Rendimiento RF sobre adult (entren+valid → prueba):", rendimiento(RF_ADULT, Xp_adult, yp_adult))
print("\n")

# -- DÍGITOS --
# Supongamos que en 4.1 hemos creado Ya:
#   X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
lista_n_arboles = [10, 20, 50]
lista_prop_muestras = [0.6, 0.8, 1.0]
lista_min_ej = [3, 5]
lista_max_prof = [5, 10]
lista_n_atrs = [50, 100, X_train_dg.shape[1]]
lista_prop_umb = [0.6, 0.8, 1.0]

mejor_comb = None
mejor_rend_val = 0.0

for n_ar in lista_n_arboles:
    for pm in lista_prop_muestras:
        for min_e in lista_min_ej:
            for mp in lista_max_prof:
                for na in lista_n_atrs:
                    for pu in lista_prop_umb:
                        rf = RandomForest(
                            n_arboles=n_ar,
                            prop_muestras=pm,
                            min_ejemplos_nodo_interior=min_e,
                            max_prof=mp,
                            n_atrs=na,
                            prop_umbral=pu
                        )
                        rf.entrena(X_train_dg, y_train_dg)
                        rend_val = rendimiento(rf, X_valid_dg, y_valid_dg)
                        if rend_val > mejor_rend_val:
                            mejor_rend_val = rend_val
                            mejor_comb = (n_ar, pm, min_e, mp, na, pu)

print("Mejor combinación en 'dígitos':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

(n_ar, pm, min_e, mp, na, pu) = mejor_comb
RF_DG = RandomForest(
    n_arboles=n_ar,
    prop_muestras=pm,
    min_ejemplos_nodo_interior=min_e,
    max_prof=mp,
    n_atrs=na,
    prop_umbral=pu
)
# Unión entrenamiento+validación
Xdg_ent_val = np.vstack((X_train_dg, X_valid_dg))
ydg_ent_val = np.hstack((y_train_dg, y_valid_dg))
RF_DG.entrena(Xdg_ent_val, ydg_ent_val)
print("Rendimiento RF sobre dígitos (entren+valid → prueba):", rendimiento(RF_DG, X_test_dg, y_test_dg))
print("\n")

# -- IMDB --
# IMDB ya viene con X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb
# Sólo necesitamos tomar parte del entrenamiento como validación:
Xe_imdb_ent, Xe_imdb_val, ye_imdb_ent, ye_imdb_val = particion_entr_prueba(X_train_imdb, y_train_imdb, test=0.2)

lista_n_arboles = [10, 20, 50]
lista_prop_muestras = [0.6, 0.8, 1.0]
lista_min_ej = [3, 5]
lista_max_prof = [5, 10]
lista_n_atrs = [50, 100, X_train_imdb.shape[1]]
lista_prop_umb = [0.6, 0.8, 1.0]

mejor_comb = None
mejor_rend_val = 0.0

for n_ar in lista_n_arboles:
    for pm in lista_prop_muestras:
        for min_e in lista_min_ej:
            for mp in lista_max_prof:
                for na in lista_n_atrs:
                    for pu in lista_prop_umb:
                        rf = RandomForest(
                            n_arboles=n_ar,
                            prop_muestras=pm,
                            min_ejemplos_nodo_interior=min_e,
                            max_prof=mp,
                            n_atrs=na,
                            prop_umbral=pu
                        )
                        rf.entrena(Xe_imdb_ent, ye_imdb_ent)
                        rend_val = rendimiento(rf, Xe_imdb_val, ye_imdb_val)
                        if rend_val > mejor_rend_val:
                            mejor_rend_val = rend_val
                            mejor_comb = (n_ar, pm, min_e, mp, na, pu)

print("Mejor combinación en 'IMDB':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

(n_ar, pm, min_e, mp, na, pu) = mejor_comb
RF_IMDB = RandomForest(
    n_arboles=n_ar,
    prop_muestras=pm,
    min_ejemplos_nodo_interior=min_e,
    max_prof=mp,
    n_atrs=na,
    prop_umbral=pu
)
# Unión entrenamiento+validación
Ximdb_ent_val = np.vstack((Xe_imdb_ent, Xe_imdb_val))
yimdb_ent_val = np.hstack((ye_imdb_ent, ye_imdb_val))
RF_IMDB.entrena(Ximdb_ent_val, yimdb_ent_val)
print("Rendimiento RF sobre IMDB (entren+valid → prueba):", rendimiento(RF_IMDB, X_test_imdb, y_test_imdb))
print("\n")








# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES NECESARIAS
# Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENEN A CONTINUACIÓN.

# EN EL APARTADO FINAL DE "RENDIMIENTOS FINALES RANDOM FOREST", USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE DEL EJERCICIO 4

# ESTE ARCHIVO trabajo_aia_23_24_parte_I.py SERÁ CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos_trabajo_aia.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS, 
# PARA EViTAR LOS ERRORES EN LA CARGA Y EJECUCIÓN.   



# *********** DESCOMENTAR A PARTIR DE AQUÍ

print("************ PRUEBAS EJERCICIO 1:")
print("**********************************\n")
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
print("\n")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
print("\n")

Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
print("\n\n\n")





print("************ PRUEBAS EJERCICIO 2:")
print("**********************************\n")

clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
clf_titanic.entrena(X_train_titanic, y_train_titanic)
clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Partido")
rend_train_titanic = rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
rend_test_titanic = rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
print(f"****** Rendimiento DT titanic train: {rend_train_titanic}")
print(f"****** Rendimiento DT titanic test: {rend_test_titanic}\n\n\n\n ")




clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
clf_votos.entrena(Xe_votos, ye_votos)
nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")
rend_train_votos = rendimiento(clf_votos,Xe_votos,ye_votos)
rend_test_votos = rendimiento(clf_votos,Xp_votos,yp_votos)
print(f"****** Rendimiento DT votos en train: {rend_train_votos}")
print(f"****** Rendimiento DT votos en test:  {rend_test_votos}\n\n\n\n")



# clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# clf_iris.entrena(X_train_iris, y_train_iris)
# clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")
# rend_train_iris = rendimiento(clf_iris,X_train_iris,y_train_iris)
# rend_test_iris = rendimiento(clf_iris,X_test_iris,y_test_iris)
# print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
# print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")





clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
clf_cancer.entrena(Xev_cancer, yev_cancer)
nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension']
clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")
rend_train_cancer = rendimiento(clf_cancer,Xev_cancer,yev_cancer)
rend_test_cancer = rendimiento(clf_cancer,Xp_cancer,yp_cancer)
print(f"***** Rendimiento DT cancer en train: {rend_train_cancer}")
print(f"***** Rendimiento DT cancer en test: {rend_test_cancer}\n\n\n")



# print("************ RENDIMIENTOS FINALES RANDOM FOREST")
# print("************************************************\n")


# # ATENCIÓN: EN CADA CASO, INCORPORAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS 
# # QUE SE HA OBTENIDO EN EL PROCESO DE AJUSTE



# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
# RF_IMDB=RandomForest(?????????????????) # ATENCIÓN: incorporar aquí los mejores valoeres de los parámetros tras el ajuste
# RF_IMDB.entrena(X_train_imdb,y_train_imdb) 
# print("Rendimiento RF entrenamiento sobre imdb: ",rendimiento(RF_IMDB,X_train_imdb,y_train_imdb))
# print("Rendimiento RF test sobre imdb: ",rendimiento(RF_IMDB,X_test_imdb,y_test_imdb))
# print("\n")




# print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")

# RF_CREDITO=RandomForest(??????????????) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_CREDITO.entrena(X_train_credito,y_train_credito) 
# print("Rendimiento RF entrenamiento sobre crédito: ",rendimiento(RF_CREDITO,X_train_credito,y_train_credito))
# print("Rendimiento RF  test sobre crédito: ",rendimiento(RF_CREDITO,X_test_credito,y_test_credito))
# print("\n")


# print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")

# RF_ADULT=RandomForest(??????????????) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
# RF_ADULT.entrena(X_train_adult,y_train_adult) 
# print("Rendimiento RF  entrenamiento sobre adult: ",rendimiento(RF_ADULT,X_train_adult,y_train_adult))
# print("Rendimiento RF  test sobre adult: ",rendimiento(RF_ADULT,X_test_adult,y_test_adult))
# print("\n")


# print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
# RF_DG=RandomForest(?????????????) # ATENCIÓN: incorporar aquí los mejores valors de losparámetros tras el ajuste
# RF_DG.entrena(X_entr_dg,y_entr_dg)
# print("Rendimiento RF entrenamiento sobre dígitos: ",rendimiento(RF_DG,X_entr_dg,y_entr_dg))
# print("Rendimiento RF validación sobre dígitos: ",rendimiento(RF_DG,X_val_dg,y_val_dg))
# print("Rendimiento RF test sobre dígitos: ",rendimiento(RF_DG,X_test_dg,y_test_dg))








