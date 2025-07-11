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
# APELLIDOS: ANDA HERNÁNDEZ
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

    n_total = X.shape[0]

    # Comrobamos que X e y coincidan con el número de muestras
    if y.shape[0] != n_total:
        raise ValueError("X e y deben tener el mismo número de filas.")

    clases_unicas, cuentas_por_clase = np.unique(y, return_counts=True)

    indices_ent = [] 
    indices_pru = []  

    # Genera índices estratificados por clase para entrenamiento y prueba
    for idx_clase, c in enumerate(clases_unicas): 
        
        indices_clase = np.where(y == c)[0].copy() 
                                                    
        np.random.shuffle(indices_clase)

        cuenta_c = cuentas_por_clase[idx_clase] 
        n_pru_c = int(np.round(cuenta_c * test)) 

        indices_pru_c = indices_clase[:n_pru_c] 
        indices_ent_c = indices_clase[n_pru_c:]

        indices_pru.extend(indices_pru_c.tolist()) 
        indices_ent.extend(indices_ent_c.tolist())

    # Baraja índices y construye los subconjuntos de datos
    indices_ent = np.array(indices_ent)
    indices_pru = np.array(indices_pru)

    np.random.shuffle(indices_ent)
    np.random.shuffle(indices_pru)

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


# Función auxiliar para calcular la ganancia de información
def entropia(y_sub): 
                    
    clases_sub, cuentas_sub = np.unique(y_sub, return_counts=True) 
    if cuentas_sub.size == 0:
        return 0.0
    p = cuentas_sub.astype(float) / cuentas_sub.sum() 
    p = p[p > 0] 
    return -np.sum(p * np.log2(p)) 


class ArbolDecision:
    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
    
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof                  = max_prof
        self.n_atrs                    = n_atrs
        self.prop_umbral               = prop_umbral
        # Raíz del árbol y lista de atributos candidatos (se inicializan en entrena)
        self.raiz = None 
        self.atributos_candidatos = None

    def entrena(self, X, y):
        
        n_ejemplos, n_atributos_totales = X.shape 

        # Si el número de atributos pasado como parámetro es menor que el total de atributos entonces elegimos aleatoriamente  atributos
        # sin sobrepasar el tamaño maximo y sin reemplazo.
        if self.n_atrs < n_atributos_totales: 
            self.atributos_candidatos = np.random.choice( 
                a       = n_atributos_totales, 
                size    = self.n_atrs, 
                replace = False 
            )
        else:
            # Sino, entrenamos el árbol con todos los atributos
            self.atributos_candidatos = np.arange(n_atributos_totales) 

        # LLamada a a la función "_construye_arbol_rec"
        self.raiz = self._construye_arbol_rec(X, y, prof=0) 

    def _construye_arbol_rec(self, X_n, y_n, prof):

       
        clases_sub, cuentas_sub = np.unique(y_n, return_counts=True)
        # Diccionario formado por cada clase y el número de muestras para cada clase 
        dict_distr_padre = {clase: int(cuenta) for clase, cuenta in zip(clases_sub, cuentas_sub)} 
        
        # Condiciones de parada (creación de hoja):
        if (prof >= self.max_prof) or (len(y_n) < self.min_ejemplos_nodo_interior) or (clases_sub.size == 1): 
            # Si se cumple una de esas condiciones de parada -> Devolvemos un nodo con el diccionario y la clase mayoritaria
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

        # Si no paramos, buscamos el MEJOR atributo A y umbral u que maximicen la ganancia de información
        mejor_gain   = -np.inf
        mejor_A      = None
        mejor_umbral = None
        H_padre      = entropia(y_n)

        # Para cada atributo candidato:
        for A in self.atributos_candidatos:
            n_nodo = X_n.shape[0] 
            # k es el número de muestras obtenido mediante la proporción del umbral que utilizamos para buscar el atributo candidato
            k = max(1, int(np.round(self.prop_umbral * n_nodo)))
            indices_muestra = np.random.choice(n_nodo, size=k, replace=False) 

            Xm = X_n[indices_muestra, A] 
            ym = y_n[indices_muestra]

            # Ordenamos las muestras para poder comparar los cambios de clase y por tanto clacular posibles umbrales para el candidato A
            orden_muestra = np.argsort(Xm) 
            Xm_orden = Xm[orden_muestra]
            ym_orden = ym[orden_muestra] 

            # Si hay un cambio de clase calculamos el corte entre los dos valores de la clase
            umbrales_A = []
            for i in range(len(Xm_orden) - 1):
                if ym_orden[i] != ym_orden[i + 1]: 
                    u = (Xm_orden[i] + Xm_orden[i + 1]) / 2.0
                    umbrales_A.append(u) 

            if len(umbrales_A) == 0: 
                continue  

            # Por cada umbral calculado para el atributo A calculamos la ganancia de información y comprobamos si es mayor que la anterior
            # hasta obtener la mejor
            for u in umbrales_A:
                mask_izq = (X_n[:, A] <= u) 
                mask_der = ~mask_izq
                if (np.sum(mask_izq) == 0) or (np.sum(mask_der) == 0):
                    continue  

                y_izq = y_n[mask_izq] 
                y_der = y_n[mask_der]

                # Calculamos la entropía para los hijos mediante la función auxiliar con las etiquetas según el umbral
                H_izq = entropia(y_izq) 
                H_der = entropia(y_der)

                N_nodo = len(y_n)
                N_izq  = len(y_izq)
                N_der  = len(y_der)

                # Calculamos la entropía de los dos hijos y se la restamos a la entropía del padre obteniendo así la ganancia de información
                # Cuanto mayor ganancia mejor ya que indica que la entropía de los hijos será pequeña y por lo tanto el corte dividirá muy bien
                # las clases según el umbral
                H_hijos = (N_izq / N_nodo) * H_izq + (N_der / N_nodo) * H_der
                gain    = H_padre - H_hijos 
                
                if gain > mejor_gain:
                    mejor_gain   = gain
                    mejor_A      = int(A)
                    mejor_umbral = float(u)
       
        # Si no hemos obtenido el mejor atributo o la ganancia es menor que 0 -> Devolvemos un nodo con el diccionario y la clase mayoritaria
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
        
        # Una vez obtenido el mejor atributo y el mejor umbral, dividimos las muestras y las etiquetas según ese umbral y las repartimos 
        # para el nodo izquierdo si <= umbral y para el nodo derecho si > umbral.
        mask_split = (X_n[:, mejor_A] <= mejor_umbral) 
        X_izq, y_izq = X_n[mask_split],    y_n[mask_split]
        X_der, y_der = X_n[~mask_split],   y_n[~mask_split]

        # Llamamos de forma recursiva a la función pero con los valores ya divididos en el nodo izquierdo y derecho.
        nodo_izq = self._construye_arbol_rec(X_izq, y_izq, prof + 1)
        nodo_der = self._construye_arbol_rec(X_der, y_der, prof + 1)

        # Por último devolvemos el nodo actual con su mejor atributo, mejor umbral, el nodo a su izquierda, el nodo a su derecha,
        # el diccionario y la clase mayoritaria es igual a None ya que no es un nodo hoja
        return Nodo(
            atributo = mejor_A,
            umbral   = mejor_umbral,
            izq      = nodo_izq,
            der      = nodo_der,
            distr    = dict_distr_padre,
            clase    = None
        )

    def clasifica(self, X): 
        
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")

        n_ejemplos = X.shape[0]
        # Creamos un array que rellenaremos con las predicciones de cada muestra
        preds = np.empty(shape=(n_ejemplos,), dtype=object)

        # Por cada muestra iniciamos el nodo raíz y vamos bajando según si la muestra es menor o mayor que el umbral hasta llegar
        # a un nodo hoja.
        for i in range(n_ejemplos):
            nodo = self.raiz 
            
            while not nodo.es_hoja(): 
                if X[i, nodo.atributo] <= nodo.umbral: 
                    nodo = nodo.izq
                else:
                    nodo = nodo.der
            # Una vez llegado al nodo hoja cogemos la clase mayoritaria de ese nodo y la añadimos al array de predicciones
            preds[i] = nodo.clase 

        return preds

    def clasifica_prob(self, x): 
        
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")

        # A diferencia de la función clasifica, aquí inicializamos el nodo raíz sólo una vez ya que buscamos 
        # las probabilidades para una sola muestra
        nodo = self.raiz

        while not nodo.es_hoja():
            if x[nodo.atributo] <= nodo.umbral: 
                nodo = nodo.izq
            else:
                nodo = nodo.der

        # Una vez hemos llegado al nodo hoja obtenemos el diccionario y calculamos la suma total de las muestras
        dist = nodo.distr 
        total = sum(dist.values())
        # Devolvemos la clase y su probabilidad calculada mediante el número de muestras de esa clase entre el número total de muestras
        return {clase: cuenta / total for clase, cuenta in dist.items()} 

    def imprime_arbol(self, nombre_atrs, nombre_clase):
        
        if self.raiz is None:
            raise ClasificadorNoEntrenado("El árbol no ha sido entrenado aún.")
        
        # LLamamos a una función auxiliar para imprimir cada nodo
        self._imprime_nodo(self.raiz, prof=0, nombre_atrs=nombre_atrs, nombre_clase=nombre_clase)

    def _imprime_nodo(self, nodo, prof, nombre_atrs, nombre_clase):
        
        # Añadimos sangría según la profundidad en la que se encuentre
        espacios = "    " * prof 

        # Si el nodo es una hoja, obtenemos el diccionario de ese nodo y mostramos el nombre de la clase, la clase mayoritaria
        # y las clases y su número de muestras respectivas según el diccionario del nodo 
        if nodo.es_hoja():
            dist_limpia = { (k.item() if isinstance(k, np.generic) else k): v 
                for k, v in nodo.distr.items() } 
            print(f"{espacios}{nombre_clase}: {nodo.clase} -- {dist_limpia}")
        # Sino, obtenemos el atributo y el umbral del nodo en el que se encuentra, imprimimos los valores en pantalla y llamamos
        # de forma recursiva a la función "imprime_nodo" pero con los nodos izquierdo y derecho 
        else:
            atr = nombre_atrs[nodo.atributo]
            um  = nodo.umbral
            print(f"{espacios}{atr} <= {um:.3f}")
            self._imprime_nodo(nodo.izq, prof + 1, nombre_atrs, nombre_clase)
            print(f"{espacios}{atr} >  {um:.3f}")
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
        
        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral

        # Lista donde iremos añadiendo los árboles que entrenemos
        self.arboles = []

    def entrena(self, X, y):
        
        n_total = X.shape[0]
        
        # Calculamos el bootstrap que utilizaremos para entrenar cada árbol
        n_bootstrap = max(1, int(round(self.prop_muestras * n_total)))

        self.arboles = []
        # Para cada árbol según el "n_arboles" de la clase "RandomForest" 
        for _ in range(self.n_arboles):
           
            # Utilizamos un subconjunto aleatorio de muestras (bootstrap)
            idxs = np.random.choice(n_total, size=n_bootstrap, replace=True)
            X_bs = X[idxs]
            y_bs = y[idxs]

            # Creamos el árbol con sus atributos por defecto
            árbol = ArbolDecision(
                min_ejemplos_nodo_interior=self.min_ejemplos_nodo_interior,
                max_prof=self.max_prof,
                n_atrs=self.n_atrs,
                prop_umbral=self.prop_umbral
            )
           
            # Entrenamos el árbol con el subconjunto de muestras y etiquetas
            árbol.entrena(X_bs, y_bs)

            # Una vez entrenado lo añadimos a la lista de árboles
            self.arboles.append(árbol)

    def clasifica(self, X):
        
        if len(self.arboles) == 0:
            raise ClasificadorNoEntrenado("El Random Forest no ha sido entrenado aún.")

        n_ejemplos = X.shape[0]
        n_trees = len(self.arboles)
        
        # Array de predicciones de cada árbol para todas las muestras
        preds_por_arbol = [árbol.clasifica(X) for árbol in self.arboles] 

        # Inicializamos array de predicciones vacío
        preds_final = np.empty(shape=(n_ejemplos,), dtype=object) 

        # Por cada muestra
        for i in range(n_ejemplos):
            # Extraemos los votos de cada árbol para esa misma muestra mediante el bucle for en el array "preds_por_arbol"
            votos = [preds_por_arbol[t][i] for t in range(n_trees)] 

            # Inicializamos un diccionario formado por la clave: clase y valor: número de muestras por esa clase
            conteo = {}
            for voto in votos:
                # Si ya ha visto esa clase en el diccionario únicamente añadimos 1 al número total
                if voto in conteo:
                    conteo[voto] += 1
                else:
                    # Si todavía no ha incluído esa clase, ponemos un uno en su posición
                    conteo[voto] = 1

            clase_pred = None
            max_cuenta = -1
            # Seleccionamos la clase con más votos
            for clase, cuenta in conteo.items():
                if cuenta > max_cuenta:
                    max_cuenta = cuenta
                    clase_pred = clase

            # Añadimos esa clase como predicción en el array de predicciones
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

# Ininicalizamos el ordinalEncoder
encoder_cred = OrdinalEncoder()

X_credito_enc = encoder_cred.fit_transform(X_credito)  

# Dividimos en 80% entrenamiento (entrenamiento + validación) y 20% prueba
Xe_cred, Xp_cred, ye_cred, yp_cred = particion_entr_prueba(X_credito_enc, y_credito, test=0.20)

# Dividimos en 75% entrenamiento y 25% validación, es decir, 60 % entrenamiento, 20% validación y 20% prueba  
Xe_cred_ent, Xe_cred_val, ye_cred_ent, ye_cred_val = particion_entr_prueba(Xe_cred, ye_cred, test=0.25)


# * X_train_adult, y_train_adult, X_test_adult, y_test_adult
#   conteniendo el AdultDataset con los atributos numéricos:

df_adult = pd.read_csv("datos/adultDataset.csv")

# Todas las muestras menos las pertenecientes a la última columna que es la variable a predecir
X_adult = df_adult.iloc[:, :-1].to_numpy()   
y_adult = df_adult.iloc[:, -1].to_numpy()   

# Misma división que en crédito
Xe_adult, Xp_adult, ye_adult, yp_adult = particion_entr_prueba(X_adult, y_adult, test=0.20)
Xe_adult_ent, Xe_adult_val, ye_adult_ent, ye_adult_val = particion_entr_prueba(Xe_adult, ye_adult, test=0.25)

# Como pide el enunciado, las primeras 4 columnas son numéricas y las restantes de tipo categórico por lo que los dividimos
X_ent_num = Xe_adult_ent[:, :4].astype(float)
X_ent_cat = Xe_adult_ent[:, 4:]

X_val_num = Xe_adult_val[:, :4].astype(float)
X_val_cat = Xe_adult_val[:, 4:]

X_test_num = Xp_adult[:, :4].astype(float)
X_test_cat = Xp_adult[:, 4:]

# Inicializamos el ordinalEncoder con "handle_unknown" y "unknown_value=-1" para que las muestras que no conozca las codifique como -1.
encoder_adult = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)

# Aplicamos la codificación
X_ent_cat_enc = encoder_adult.fit_transform(X_ent_cat)

X_val_cat_enc  = encoder_adult.transform(X_val_cat)
X_test_cat_enc = encoder_adult.transform(X_test_cat)

# Por último, unimos las columnas numéricas y categóricas de cada conjunto de datos mediante el método "np.hstack" que une las columnas 
# horizontalmente
X_train_adult = np.hstack([X_ent_num, X_ent_cat_enc])
y_train_adult = ye_adult_ent

X_valid_adult = np.hstack([X_val_num, X_val_cat_enc])
y_valid_adult = ye_adult_val

X_test_adult  = np.hstack([X_test_num, X_test_cat_enc])
y_test_adult  = yp_adult


# * X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
#   conteniendo el dataset de los dígitos escritos a mano:

# Para preprocesar las imágenes de digits hemos definido dos funciones: 
    # La primera procesa las imágenes y devuelve un array [i, 28*28] donde i es cada imagen con su vector de 1D aplanado con 1s y 0s
    # La segunda devuelve un array con todas las etiquetas

# Como parámetros le pasamos el path de las imágenes ya sea de entrenamiento, validación o prueba y el número de ejemplos de cada una
def lee_imagenes_digitos(path_imagenes, n_ejemplos):

    # Creamos un array de 0s con filas = n_ejemplos y el tamañano de cada ejemplo = 28*28 (tamaño de cada imagen)
    X = np.zeros((n_ejemplos, 28*28), dtype=int)

    # Abrimos el path de las imagenes
    with open(path_imagenes, 'r') as f:
        # Por cada imagen
        for i in range(n_ejemplos):
            # Inicializamos una lista de pixeles  
            pixeles = []
            # Por cada fila de la imagen (en total 28)
            for _ in range(28): 
                # Leemos la línea
                linea = f.readline() 

                if not linea:
                    raise ValueError(f"El fichero {path_imagenes} terminó antes de leer 28 líneas para la imagen {i}.")
                
                # La línea leída la pasamos a la variable fila pero eliminando los saltos de línea mediante el método "rstrip"
                fila = linea.rstrip("\n")
                # Una vez tenemos la fila sin saltos de línea, la recorremos con un bucle y sustituímos por un 1
                #  si se encuentra con un carácter que no es un espacio en blanco (+ o *) o un 0 si es un espacio en blanco. 
                # Por tanto la lista "pixeles" tendrá 728 caracteres que serán 1 o 0, es decir, la imagen aplanada.
                pixeles.extend([0 if ch == ' ' else 1 for ch in fila]) 
                # Por último, añadimos al array X de la posición de la imagen i, el vector de 1D con todos los 1s y 0s.
            X[i, :] = np.array(pixeles, dtype=int) 
    return X

# Ésta función devuelve un array de etiquetas a partir de un path de etiquetas
def lee_labels_digitos(path_labels):
    with open(path_labels, 'r') as f:
        # Por cada línea convierte el carácter que haya en un entero y borra todos los saltos de línea, tabulaciones, etc mediante el 
        # método "strip"
        y = [int(l.strip()) for l in f.readlines()]                                                    
    return np.array(y, dtype=int) 

ruta_train_labels = "datos/digitdata/traininglabels"
ruta_valid_labels = "datos/digitdata/validationlabels"
ruta_test_labels  = "datos/digitdata/testlabels"

# LLamamos a la función para que nos devuelva el array de etiquetas
y_train_digitos = lee_labels_digitos(ruta_train_labels)
y_valid_digitos = lee_labels_digitos(ruta_valid_labels)
y_test_digitos  = lee_labels_digitos(ruta_test_labels)

n_train_d = y_train_digitos.shape[0]
n_valid_d = y_valid_digitos.shape[0]
n_test_d  = y_test_digitos.shape[0]

# LLamamos a la función para obtener el array X
X_train_digitos = lee_imagenes_digitos("datos/digitdata/trainingimages",   n_train_d)
X_valid_digitos = lee_imagenes_digitos("datos/digitdata/validationimages", n_valid_d)
X_test_digitos  = lee_imagenes_digitos("datos/digitdata/testimages",       n_test_d)

X_train_dg, y_train_dg = X_train_digitos,  y_train_digitos
X_valid_dg, y_valid_dg = X_valid_digitos,  y_valid_digitos
X_test_dg,  y_test_dg  = X_test_digitos,   y_test_digitos







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

#-- CRÉDITO --

# Dividimos en entrenamiento, validación y prueba con la función "particion_entr_prueba"

# Xe_cred, Xp_cred, ye_cred, yp_cred = particion_entr_prueba(X_credito_enc, y_credito, test=0.20)
# Xe_cred_ent, Xe_cred_val, ye_cred_ent, ye_cred_val = particion_entr_prueba(Xe_cred, ye_cred, test=0.25)

# Definimos el grid de valores para calcular la mejor combinación

# lista_n_arboles     = [10, 20]
# lista_prop_muestras = [0.6, 0.8, 1.0]
# lista_min_ej        = [3, 5, 10]
# lista_max_prof      = [5, 10]
# lista_n_atrs        = [5, 10, X_credito_enc.shape[1]]
# lista_prop_umb      = [0.6, 0.8, 1.0]

# mejor_comb         = None
# mejor_rend_val     = 0.0

# Gridsearch con todos los valores posibles

# for n_ar in lista_n_arboles:
#     for pm in lista_prop_muestras:
#         for min_e in lista_min_ej:
#             for mp in lista_max_prof:
#                 for na in lista_n_atrs:
#                     for pu in lista_prop_umb:
#                         rf = RandomForest(
#                             n_arboles=n_ar,
#                             prop_muestras=pm,
#                             min_ejemplos_nodo_interior=min_e,
#                             max_prof=mp,
#                             n_atrs=na,
#                             prop_umbral=pu
#                         )
#                         # Por cada árbol con los diferentes valores calculamos el rendimiento y elegimos el que devuelva el mejor
#                         rf.entrena(Xe_cred_ent, ye_cred_ent)
#                         rend_val = rendimiento(rf, Xe_cred_val, ye_cred_val)
#                         if rend_val > mejor_rend_val:
#                             mejor_rend_val = rend_val
#                             mejor_comb = (n_ar, pm, min_e, mp, na, pu)

# print("Mejor combinación en 'crédito':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

# # Reentreno definitivo sobre entrenamiento+validación
# (n_ar, pm, min_e, mp, na, pu) = mejor_comb
# RF_CREDITO = RandomForest(
#     n_arboles=n_ar,
#     prop_muestras=pm,
#     min_ejemplos_nodo_interior=min_e,
#     max_prof=mp,
#     n_atrs=na,
#     prop_umbral=pu
# )
# # Unimos el entrenamiento y validación para entrenar de forma definitiva el mejor clasificador
# Xcred_ent_val = np.vstack((Xe_cred_ent, Xe_cred_val))
# ycred_ent_val = np.hstack((ye_cred_ent, ye_cred_val))
# RF_CREDITO.entrena(Xcred_ent_val, ycred_ent_val)
# print("Rendimiento RF sobre crédito (entren+valid → prueba):", rendimiento(RF_CREDITO, Xp_cred, yp_cred))
# print("\n")

#  TODOS LOS SIGUIENTES ENTRENAMIENTOS DE DATASETS FORMAN LA MISMA ESTRUCTURA CON GRIDSEARCH CON VALIDACIÓN

# # -- ADULTDATASET --

# # Usamos X_adult_enc 
# Xe_adult, Xp_adult, ye_adult, yp_adult = particion_entr_prueba(X_adult_enc, y_adult, test=0.30)
# Xe_adult_ent, Xe_adult_val, ye_adult_ent, ye_adult_val = particion_entr_prueba(Xe_adult, ye_adult, test=0.25)


# lista_n_arboles     = [10, 20]
# lista_prop_muestras = [0.6, 0.8, 1.0]
# lista_min_ej        = [5, 10]
# lista_max_prof      = [5, 10]
# lista_n_atrs        = [5, 10, X_adult.shape[1]]
# lista_prop_umb      = [0.6, 0.8, 1.0]

# mejor_comb     = None
# mejor_rend_val = 0.0

# for n_ar in lista_n_arboles:
#     for pm in lista_prop_muestras:
#         for min_e in lista_min_ej:
#             for mp in lista_max_prof:
#                 for na in lista_n_atrs:
#                     for pu in lista_prop_umb:
#                         rf = RandomForest(
#                             n_arboles=n_ar,
#                             prop_muestras=pm,
#                             min_ejemplos_nodo_interior=min_e,
#                             max_prof=mp,
#                             n_atrs=na,
#                             prop_umbral=pu
#                         )
#                         rf.entrena(Xe_adult_ent, ye_adult_ent)
#                         rend_val = rendimiento(rf, Xe_adult_val, ye_adult_val)
#                         if rend_val > mejor_rend_val:
#                             mejor_rend_val = rend_val
#                             mejor_comb = (n_ar, pm, min_e, mp, na, pu)

# print("Mejor combinación en 'adult':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

# (n_ar, pm, min_e, mp, na, pu) = mejor_comb
# RF_ADULT = RandomForest(
#     n_arboles=n_ar,
#     prop_muestras=pm,
#     min_ejemplos_nodo_interior=min_e,
#     max_prof=mp,
#     n_atrs=na,
#     prop_umbral=pu
# )
# Xadult_ent_val = np.vstack((Xe_adult_ent, Xe_adult_val))
# yadult_ent_val = np.hstack((ye_adult_ent, ye_adult_val))
# RF_ADULT.entrena(Xadult_ent_val, yadult_ent_val)
# print("Rendimiento RF sobre adult (entren+valid → prueba):", rendimiento(RF_ADULT, Xp_adult, yp_adult))
# print("\n")

# # -- DÍGITOS --

# X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
# lista_n_arboles = [10, 20]
# lista_prop_muestras = [0.6, 0.8, 1.0]
# lista_min_ej = [3, 5]
# lista_max_prof = [5, 10]
# lista_n_atrs = [50, 100, X_train_dg.shape[1]]
# lista_prop_umb = [0.6, 0.8, 1.0]

# mejor_comb = None
# mejor_rend_val = 0.0

# for n_ar in lista_n_arboles:
#     for pm in lista_prop_muestras:
#         for min_e in lista_min_ej:
#             for mp in lista_max_prof:
#                 for na in lista_n_atrs:
#                     for pu in lista_prop_umb:
#                         rf = RandomForest(
#                             n_arboles=n_ar,
#                             prop_muestras=pm,
#                             min_ejemplos_nodo_interior=min_e,
#                             max_prof=mp,
#                             n_atrs=na,
#                             prop_umbral=pu
#                         )
#                         rf.entrena(X_train_dg, y_train_dg)
#                         rend_val = rendimiento(rf, X_valid_dg, y_valid_dg)
#                         if rend_val > mejor_rend_val:
#                             mejor_rend_val = rend_val
#                             mejor_comb = (n_ar, pm, min_e, mp, na, pu)

# print("Mejor combinación en 'dígitos':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

# (n_ar, pm, min_e, mp, na, pu) = mejor_comb
# RF_DG = RandomForest(
#     n_arboles=n_ar,
#     prop_muestras=pm,
#     min_ejemplos_nodo_interior=min_e,
#     max_prof=mp,
#     n_atrs=na,
#     prop_umbral=pu
# )
# # Unión entrenamiento+validación
# Xdg_ent_val = np.vstack((X_train_dg, X_valid_dg))
# ydg_ent_val = np.hstack((y_train_dg, y_valid_dg))
# RF_DG.entrena(Xdg_ent_val, ydg_ent_val)
# print("Rendimiento RF sobre dígitos (entren+valid → prueba):", rendimiento(RF_DG, X_test_dg, y_test_dg))
# print("\n")

# # -- IMDB --

# # IMDB ya viene con X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb
# # Sólo necesitamos tomar parte del entrenamiento como validación:
# Xe_imdb_ent, Xe_imdb_val, ye_imdb_ent, ye_imdb_val = particion_entr_prueba(X_train_imdb, y_train_imdb, test=0.2)

# lista_n_arboles = [10, 20]
# lista_prop_muestras = [0.6, 0.8, 1.0]
# lista_min_ej = [3, 5]
# lista_max_prof = [5, 10]
# lista_n_atrs = [50, 100, X_train_imdb.shape[1]]
# lista_prop_umb = [0.6, 0.8, 1.0]

# mejor_comb = None
# mejor_rend_val = 0.0

# for n_ar in lista_n_arboles:
#     for pm in lista_prop_muestras:
#         for min_e in lista_min_ej:
#             for mp in lista_max_prof:
#                 for na in lista_n_atrs:
#                     for pu in lista_prop_umb:
#                         rf = RandomForest(
#                             n_arboles=n_ar,
#                             prop_muestras=pm,
#                             min_ejemplos_nodo_interior=min_e,
#                             max_prof=mp,
#                             n_atrs=na,
#                             prop_umbral=pu
#                         )
#                         rf.entrena(Xe_imdb_ent, ye_imdb_ent)
#                         rend_val = rendimiento(rf, Xe_imdb_val, ye_imdb_val)
#                         if rend_val > mejor_rend_val:
#                             mejor_rend_val = rend_val
#                             mejor_comb = (n_ar, pm, min_e, mp, na, pu)

# print("Mejor combinación en 'IMDB':", mejor_comb, "con rendimiento en validación:", mejor_rend_val)

# (n_ar, pm, min_e, mp, na, pu) = mejor_comb
# RF_IMDB = RandomForest(
#     n_arboles=n_ar,
#     prop_muestras=pm,
#     min_ejemplos_nodo_interior=min_e,
#     max_prof=mp,
#     n_atrs=na,
#     prop_umbral=pu
# )
# # Unión entrenamiento+validación
# Ximdb_ent_val = np.vstack((Xe_imdb_ent, Xe_imdb_val))
# yimdb_ent_val = np.hstack((ye_imdb_ent, ye_imdb_val))
# RF_IMDB.entrena(Ximdb_ent_val, yimdb_ent_val)
# print("Rendimiento RF sobre IMDB (entren+valid → prueba):", rendimiento(RF_IMDB, X_test_imdb, y_test_imdb))
# print("\n")

# ----------------------------

# A continuación se describen las mejores combinaciones encontradas para cada dataset:

# ── CRÉDITO ────────────────────────────────────────
# Combinaciones probadas (ejemplos):
# lista_n_arboles = [10, 20]
# lista_prop_muestras = [0.8, 1.0]
# lista_min_ej = [5, 10]
# lista_max_prof = [10]
# lista_n_atrs = [6, 10]
# lista_prop_umb = [0.8, 1.0]

# Mejor combinación obtenida:
# (n_arboles=10, prop_muestras=0.8, min_ejemplos=5, max_prof=10, n_atrs=10, prop_umbral=0.8)
# Rendimiento en validación: 0.9461538461538461
# Rendimiento final en test: 0.9153846153846154
# ----------------------------

# ── ADULTDATASET ────────────────────────────────────────
# lista_n_arboles     = [5]
# lista_prop_muestras = [1.0]
# lista_min_ej        = [10]
# lista_max_prof      = [10]
# lista_n_atrs        = [10]
# lista_prop_umb      = [0.8, 1.0]

# Mejor combinación obtenida:
# (n_arboles=5, prop_muestras=1.0, min_ejemplos=10, max_prof=10, n_atrs=10, prop_umbral=1.0)
# Rendimiento en validación: 0.8494208494208494
# Rendimiento final en test: 0.8523751023751024
# ----------------------------

# ── DÍGITOS ────────────────────────────────────────
# Combinaciones probadas (ejemplos):
# lista_n_arboles = [10]
# lista_prop_muestras = [1.0]
# lista_min_ej = [3]
# lista_max_prof = [5]
# lista_n_atrs = [50]
# lista_prop_umb = [1.0]


# Mejor combinación obtenida:
# (n_arboles=20, prop_muestras=0.8, min_ejemplos=3, max_prof=5, n_atrs=50, prop_umbral=0.8)
# Rendimiento en validación: 0.787
# Rendimiento final en test: 0.748
# ----------------------------

# ── IMDB ────────────────────────────────────────
# Combinaciones probadas (ejemplos):
# lista_n_arboles = [5, 10]
# lista_prop_muestras = [0.8, 1.0]
# lista_min_ej = [3, 5]
# lista_max_prof = [5, 10]
# lista_n_atrs = [80]
# lista_prop_umb = [0.8, 1.0]
#
# Mejor combinacion obtenida:
# (n_arboles=10, prop_muestras=0.8, min_ejemplos=3, max_prof=10, n_atrs=60, prop_umbral=1.0)
# Rendimiento en validación: 0.6925
# Rendimiento final en test: 0.69
#
# ----------------------------


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



# # *********** DESCOMENTAR A PARTIR DE AQUÍ

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

# Hemos añadido la partición en entreamiento y prueba del conjunto de datos Iris
X_train_iris, X_test_iris, y_train_iris, y_test_iris = particion_entr_prueba(X_iris, y_iris, test=0.2)
print("Estratificación entrenamiento iris: ",np.unique(y_train_iris,return_counts=True))
print("Estratificación prueba iris: ",np.unique(y_test_iris,return_counts=True))
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



clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
clf_iris.entrena(X_train_iris, y_train_iris)
clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")
rend_train_iris = rendimiento(clf_iris,X_train_iris,y_train_iris)
rend_test_iris = rendimiento(clf_iris,X_test_iris,y_test_iris)
print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")



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



print("************ RENDIMIENTOS FINALES RANDOM FOREST")
print("************************************************\n")


# ATENCIÓN: EN CADA CASO, INCORPORAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS 
# QUE SE HA OBTENIDO EN EL PROCESO DE AJUSTE



print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
RF_IMDB=RandomForest(n_arboles=10, prop_muestras=0.8, min_ejemplos_nodo_interior=3, max_prof=10, n_atrs=60, prop_umbral=1.0) # ATENCIÓN: incorporar aquí los mejores valoeres de los parámetros tras el ajuste
RF_IMDB.entrena(X_train_imdb,y_train_imdb) 
print("Rendimiento RF entrenamiento sobre imdb: ",rendimiento(RF_IMDB,X_train_imdb,y_train_imdb))
print("Rendimiento RF test sobre imdb: ",rendimiento(RF_IMDB,X_test_imdb,y_test_imdb))
print("\n")



print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")

RF_CREDITO=RandomForest(n_arboles=10, prop_muestras=0.8, min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=0.8) 
RF_CREDITO.entrena(Xe_cred_ent, ye_cred_ent) 
print("Rendimiento RF entrenamiento sobre crédito: ",rendimiento(RF_CREDITO,Xe_cred_ent, ye_cred_ent))
print("Rendimiento RF  test sobre crédito: ",rendimiento(RF_CREDITO,Xp_credito, yp_credito))
print("\n")


print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")

RF_ADULT=RandomForest(n_arboles=5, prop_muestras=1.0, min_ejemplos_nodo_interior=10, max_prof=10, n_atrs=10, prop_umbral=1.0) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
RF_ADULT.entrena(X_train_adult,y_train_adult) 
print("Rendimiento RF  entrenamiento sobre adult: ",rendimiento(RF_ADULT,X_train_adult,y_train_adult))
print("Rendimiento RF  test sobre adult: ",rendimiento(RF_ADULT,X_test_adult,y_test_adult))
print("\n")


print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
RF_DG=RandomForest(n_arboles=20, prop_muestras=0.8, min_ejemplos_nodo_interior=3, max_prof=5, n_atrs=50, prop_umbral=0.8) # ATENCIÓN: incorporar aquí los mejores valors de losparámetros tras el ajuste
RF_DG.entrena(X_train_dg, y_train_dg)
print("Rendimiento RF entrenamiento sobre dígitos: ",rendimiento(RF_DG,X_train_dg, y_train_dg))
print("Rendimiento RF validación sobre dígitos: ",rendimiento(RF_DG,X_valid_dg,y_valid_dg))
print("Rendimiento RF test sobre dígitos: ",rendimiento(RF_DG,X_test_dg,y_test_dg))








