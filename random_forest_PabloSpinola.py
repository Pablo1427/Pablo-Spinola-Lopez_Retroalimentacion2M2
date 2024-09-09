# @author: Pablo Spínola López
#          A01753922
# Descripción: A continuación, se presenta un programa dedicado a entrenar un modelo de clasificación con random forest aportado por la linrería de SciKit Learn.
#              Posterior al entrenamiento, se visualizan gráficas de desempeño y métricas de exactitud tras realizar predicciones con datos no vistos durante la fase del entrenamiento.


# numpy para convertir los datasets a matrices y poderlos utilizar para las predicciones.
from numpy import array
# Tanto matplotlib como seaborn son utilizados para construir y visualizar la matriz de confusión como un heatmap.
import matplotlib.pyplot as plt
from seaborn import heatmap
### SciKit Learn para herramientas de machine learning, como división y evaluación, y el modelo de Random Forest.
#    Se importan las funciones que nos permiten evaluar el rendimiento y exactitud del modelo, como el reporte de clasificación, la exactitud y matriz de confusión.
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#    Se importa el Random Forest Classifier, otorgando la capacidad de entrenar un modelo de clasificación con random forest.
from sklearn.ensemble import RandomForestClassifier
#    Se importa la función para dividir un dataset dado en entrenamiento y testeo según las variables dadas, la proporción de los subsets
#    resultantes y una semilla de aleatoriedad para reacomodar los datos y asegurar el "desorden".
from sklearn.model_selection import train_test_split


# Datos proporcionados para el dataset a evaluar
# 'var_indep' es la lista de valores independientes que usaremos para entrenar el modelo.
# 'var_dep' es la lista de valores dependientes que se clasifican la variable independiente si es >= 70 o no:
#       Si la variable independiente es mayor o igual a 70, se clasifica con 1 (examen aprobado).
#       Si la variable independiente es menor que 70, se clasifica con 0 (examen reaprobado).
var_indep = [ 45, 78, 12, 134,  56, 89, 23, 167,  90,  3, 145, 67, 29, 101, 76, 58, 172, 33, 110,  5, 
             149, 62, 84,  19, 138, 72, 41, 160,  94,  8, 153, 69, 27, 115, 81, 50, 175, 36, 123, 11,
             147, 64, 87,  21, 141, 74, 43, 163,  97, 14, 156, 71, 31, 118, 83, 52, 178, 39, 126, 17,
             151, 66, 92,  25, 143, 77, 48, 165,  99, 20, 158, 73, 35, 121, 85, 54, 180, 42, 128, 22,
             155, 68, 95,  28, 140, 79, 46, 169, 102, 24, 161, 75, 38, 130, 88, 57, 177, 44, 132, 30]
# Crea una lista de 1 o 0 dependiendo de la condición anterior; menor o mayor que 70.
var_dep = [1 if num >= 70 else 0 for num in var_indep]

### Dividisión del dataset completo en entrenamiento, validación y testeo
#    Se divide el dataset en set de entrenamiento, validación y testeo.
#    Primero, se separa el 60% de los datos para entrenamiento, para que el 40% se divida posteriormente en validación y testeo.
X_train, X_temp, y_train, y_temp = train_test_split(
    var_indep,
    var_dep,
    test_size=0.4,  # El 40% de los datos seran para validación y testeo, es decir, el 60% son para entrenameinto.
    random_state=35
)
#    Una vez que se han almacenado el 40% de los datos en X_temo y y_temp, dividimos este total en 2 partes:
#       - La mitad para validación: 50% de este conjunto (20% del total inicial)
#       - La mitad para testeo: 50% de este conjunto (20% del total inicial)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=35
)

# Una vez que se tienen los subsets para el entrenamiento, los conjuntos unidimensionales que contienen las variables independientes
# se transforman en matrices, ya que la funcion RandomForestClassifier requiere que las listas sean de 2 dimensiones.
X_train_mat = array(X_train).reshape(-1, 1)
X_val_mat = array(X_val).reshape(-1, 1)
X_test_mat = array(X_test).reshape(-1, 1)

# Entrenar el modelo de Random Forest
# Se crea y entrena el modelo de Random Forest usando el conjunto de entrenamiento.
random_forest = RandomForestClassifier(random_state=35)
random_forest.fit(X_train_mat, y_train)

# Una vez entrenado el modelo, se realizan predicciones con los conjuntos de validación y testeo para medir la precisión adquirida en el entrenamiento.
y_val_pred = random_forest.predict(X_val_mat)
y_test_pred = random_forest.predict(X_test_mat)

### Evaluar el rendimiento y precisión del modelo al predecir los conjuntos de validación y testeo
#    Se muestra un reporte del rendimiento que incluye métricas para evaluar el modelo, tales como la precisión, recall, F1-score y soporte.
print("Métricas para evaluar con subset de Validación:")
print(classification_report(y_val, y_val_pred))
print("Métricas para evaluar con subset de Test:")
print(classification_report(y_test, y_test_pred))
#    Se muestran los porcentajes de exactitud que mostró el modelo al momento de predecir las variables de subset de validación y de testeo.
exactitud_val = accuracy_score(y_val, y_val_pred)
exactitud_test = accuracy_score(y_test, y_test_pred)
print(f"Precisión con subset de Validación: {(exactitud_val*100):.2f}%")
print(f"Precisión con subset de Test: {(exactitud_test*100):.2f}%")

# Matriz de confusión
# Se muestra la matriz de confusión de forma gráfica con la finalidad de evaluar el desempeño del modelo con el subconjunto de testeo y visualizar los aciertos de la predicción.
conf_mat = confusion_matrix(y_test, y_test_pred)
heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho 0', 'Predicho 1'], yticklabels=['Real 0', 'Real 1'])
plt.title('Matriz de Confusión con subset de Test')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()


### Una vez evaluado el modelo, se realizan predicciones con datos no visto durante el entrenamiento.
#    Se crea un nuevo dataset con valores desconocidos para el modelo, incluyendo datos incluso menores que 0 y mayores de 180, como -49 o 350.
#    También se encuentra el valor de referencia para la clasificación: 70, para saber si es que el modelo identificó esta clasificación
eval_data_indep = [-15, -14, -10, -49, -48, -22, -41, -34, -26, -38,
                    -6, -33, -27, -46, -29, -17, -31, -19, -28, -40,
                   -11, -36, -32, -24, -50, -23,  -7, -12, -43, -44,
                     0,   6, 100, 179,  66, 188, 200, 230, -99, 350,
                    -1, 181, 194, 186,  55,  51,  70,  69,  55, 144]
real_Y = [1 if num >= 70 else 0 for num in eval_data_indep]


#    Se vuelve a hacer la conversión de la lista de valores independientes a matriz.
eval_data_indep_mat = array(eval_data_indep).reshape(-1, 1)


#    Se calculan predicciones del modelo para los nuevos datos.
predicciones = random_forest.predict(eval_data_indep_mat)

### Comparación de las predicciones con los valores reales
print("\n\n\n")
#    Se calcula nuevamente el porcentaje de exactitud para este nuevo conjunto de datos.
exactitud_real = accuracy_score(real_Y, predicciones)
print(f"Porcentaje de precisión con valores antes no vistos: {(exactitud_real*100):.2f}%")

# Nuevamente se muestra una matriz de confusión con la evaluación de las predicciones con los datos nuevos para el modelo.
conf_mat = confusion_matrix(real_Y, predicciones)
heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho 0', 'Predicho 1'], yticklabels=['Real 0', 'Real 1'])
plt.title('Matriz de Confusión con set final')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()

#    Se muestran los valores indepentientes junto con sus predicciones y el valor real, con la finalidad de ver cuáles fueron los valores correctamente clasificados
#    e identificar aquellos valores mostrados en la matriz de confusión.
#    Se muestra True en caso de que se haya clasificado correctamente.
print("\nPredicciones vs Valores Reales:")
for i in range(len(eval_data_indep)):
    print(f"Para: {eval_data_indep[i]}, Predicción: {predicciones[i]}, Real: {real_Y[i]}:   {predicciones[i] == real_Y[i]}")
