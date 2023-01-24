import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
# import matplotlib.pyplot as plt

# Cargar los datos en un DataFrame
df = pd.read_csv('delivery_dataset.csv', sep=';')

# Limpiar y preparar los datos
df = df.dropna() # eliminar valores nulos
df = df.drop_duplicates() # eliminar duplicados

# Crear variable dependiente y variable independiente
x = df[['Actual_Shipment_Time', 'Shipment_Delay', 'Distance']]
y = df['Delivery_Status']

x_train = x.head(4000)
y_train = y.head(4000)
svm = SVC(kernel = 'rbf', probability=True, gamma='scale')

svm.fit(x_train,y_train)

x_pred = x.iloc[5000]

output = svm.predict([x_pred])
print(y.iloc[5000])
print(svm.classes_)
print("probabilities:", svm.predict_proba([x_pred]))