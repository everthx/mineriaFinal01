"""import pandas as pd
from pandas import DataFrame # Capturar datos
from sklearn.cluster import KMeans # Aplicar el K-Means Clustering
import matplotlib.pyplot as plt # Crear graficos

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

# Carga del conjunto de datos
dataset = pd.read_csv('POBLACION_MF - TOTAL.csv', encoding="ISO-8859-1")

#print(dataset)

df = DataFrame(dataset, columns=['HOMBRES', 'MUJERES'])
Data = {'HOMBRES':[142279,152154,154548,151811,151887,135511,129848,124218,126874,100569,87536,61735,47479,313,22559,29088,945],
        'MUJERES':[138.358,148176,151999,146153,151303,142492,130998,129856,124648,98393,86845,65284,53417,38222,2369,34882,709]}
df = DataFrame(Data, columns=['HOMBRES','MUJERES'])

#print(df)

#Ejemplo con 3 clustering

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['HOMBRES'], df['MUJERES'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
"""


import pandas as pd
from datafrane import Dataframe

pd.set_option('display.width',300)
pd.set_option('display.max_columns',50)
df = pd.read_csv('01_POBLACION_01_BC.csv', encoding="ISO-8859-1")

nuevoDataset = Dataframe.filtrar_dataset(df)
print(nuevoDataset)

