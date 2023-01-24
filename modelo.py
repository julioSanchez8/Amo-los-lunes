import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt

# Cargar los datos en un DataFrame
df = pd.read_csv('delivery_dataset.csv', sep=';')

# Limpiar y preparar los datos
df = df.dropna() # eliminar valores nulos
df = df.drop_duplicates() # eliminar duplicados

# Crear variable dependiente y variable independiente
x = df[['Actual_Shipment_Time', 'Shipment_Delay', 'Distance']]
y = df['Delivery_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
svm = SVC(kernel = 'linear', probability=True, gamma='scale')

svm.fit(x_train,y_train)

output = svm.predict(x_test)


print("exactitud:",metrics.accuracy_score(y_test, output))
print("probabilidades:", svm.predict_proba(x_test))
print("Precision:",metrics.precision_score(y_test, output))

