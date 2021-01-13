import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans

import subprocess

class Dataframe():

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('01_POBLACION_01_BC.csv', encoding="ISO-8859-1")

    def filtrar_dataset(df):
        df = df.rename(columns={'INEGI. Tabulados de la Encuesta Intercensal 2015': 'Estado'
            , 'Unnamed: 1': 'Dimension', 'Unnamed: 2': 'Edades', 'Unnamed: 3': 'Estimador',
                                'Unnamed: 4': 'Poblacion Total'
            , 'Unnamed: 5': 'Hombres', 'Unnamed: 6': 'Mujeres'})
        df = df.loc[7: 654].reset_index(drop=True)
        df = df.loc[df['Estimador'] == 'Valor'].reset_index(drop=True)

        #       Convertir columnas a valor entero
        df['Poblacion Total'] = df['Poblacion Total'].str.replace(',', '')
        df['Hombres'] = df['Hombres'].str.replace(',', '')
        df['Mujeres'] = df['Mujeres'].str.replace(',', '')
        df['Poblacion Total'] = df['Poblacion Total'].astype(int)
        df['Hombres'] = df['Hombres'].astype(int)
        df['Mujeres'] = df['Mujeres'].astype(int)

        #       Guardar nuevo dataset filtrado
        df.to_csv('poblacion_filtrado.csv', index=False)
        df = pd.read_csv('poblacion_filtrado.csv')
        return df

    def filtrar_ciudades(df):
        #       Relacionar ciudades por su rango de habitantes
        df['Dimension'] = df['Dimension'].str.replace('Menos de 2 500 habitantes', 'Rosarito')
        df['Dimension'] = df['Dimension'].str.replace('2 500-14 999 habitantes', 'Tecate')
        df['Dimension'] = df['Dimension'].str.replace('15 000-49 999 habitantes', 'Ensenada')
        df['Dimension'] = df['Dimension'].str.replace('50 000-99 999 habitantes', 'Mexicali')
        df['Dimension'] = df['Dimension'].str.replace('100 000 y más habitantes', 'Tijuana')

        #       Cambio a categorias numericas los rangos de edades
        df['Edades'] = df['Edades'].str.replace('00-04 años', '1')
        df['Edades'] = df['Edades'].str.replace('05-09 años', '2')
        df['Edades'] = df['Edades'].str.replace('10-14 años', '3')
        df['Edades'] = df['Edades'].str.replace('15-19 años', '4')
        df['Edades'] = df['Edades'].str.replace('20-24 años', '5')
        df['Edades'] = df['Edades'].str.replace('25-29 años', '6')
        df['Edades'] = df['Edades'].str.replace('30-34 años', '7')
        df['Edades'] = df['Edades'].str.replace('35-39 años', '8')
        df['Edades'] = df['Edades'].str.replace('40-44 años', '9')
        df['Edades'] = df['Edades'].str.replace('45-49 años', '10')
        df['Edades'] = df['Edades'].str.replace('50-54 años', '11')
        df['Edades'] = df['Edades'].str.replace('55-59 años', '12')
        df['Edades'] = df['Edades'].str.replace('60-64 años', '13')
        df['Edades'] = df['Edades'].str.replace('65-69 años', '14')
        df['Edades'] = df['Edades'].str.replace('70-74 años', '15')
        df['Edades'] = df['Edades'].str.replace('75 años y más', '16')
        df['Edades'] = df['Edades'].str.replace('No especificado', '0')
        df['Edades'] = df['Edades'].str.replace('Total', '17')

        #       Convertir columna Edades a tipo Entero
        df['Edades'] = df['Edades'].astype(int)

        #       Guardar a CSV
        df.to_csv('poblacion_ciudades.csv', index=False)
        df = pd.read_csv('poblacion_ciudades.csv')
        return df

    def plot_habitantes_municipio(df):
        #       Separar a columnas necesarias para el Plot
        ciudades = df.loc[(~df['Dimension'].str.contains('Total')) & (df['Edades'] == 17)]
        a = (ciudades['Poblacion Total'])
        b = (ciudades['Dimension'])

        #       Haciendo el Plot
        plt.figure('Gráfica: Habitantes por Municipio')
        plt.pie(a, labels=b, autopct="%0.1f %%", startangle=140)
        plt.axis("equal")
        plt.title('Porcentaje de Habitantes por municipio en B.C.\n\n')
        plt.show()

    def plot_sexo_municipio(df):
        #       Separar a columnas necesarias para el Plot
        df = df.loc[(~df['Dimension'].str.contains('Total')) & (df['Edades'] == 17)]
        df = df.drop(['Estado','Edades','Estimador','Poblacion Total'], axis=1)
        df['%Hombres'] = (df['Hombres'] / (df['Hombres'] + df['Mujeres'])) * 100
        df['%Mujeres'] = (df['Mujeres'] / (df['Hombres'] + df['Mujeres'])) * 100
        etiquetas, y1, y2 = (df['Dimension'], df['%Hombres'], df['%Mujeres'])

        plt.figure('Gráfica: Composicón por Municipio segun su Sexo')
        X = np.arange(5)
        pl.bar(X, +y1, color='cornflowerblue', edgecolor='white')
        pl.bar(X, -y2, color='pink', edgecolor='white')

        #       Desplega porcentajes para Hombres y Mujeres
        for x, y in zip(X, y1):
            pl.text(x + 0.1, y + 0.025, '%.2f ' % y + '%', ha='center', va='bottom')
        for x, y in zip(X, y2):
            pl.text(x + 0.1, -y + 0.025, '%.2f' % y + '%', ha='center', va='top')

        #       Opciones de desplegado y PLOT
        pl.title('Composición por Municipio segun su Sexo')
        pl.legend(labels=['Hombres', 'Mujeres'])
        pl.xticks(np.arange(5), etiquetas)
        pl.ylabel('Porcentaje por sexo')
        pl.ylim(-60.0, +80.0)
        plt.show()

    def plot_edades_total(df):
        #       Separar a columnas necesarias para el Plot
        df = df.loc[(df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Estimador', 'Poblacion Total', 'Dimension'], axis=1)
        df['%Hombres'] = (df['Hombres'] / (df['Hombres'] + df['Mujeres'])) * 100
        df['%Mujeres'] = (df['Mujeres'] / (df['Hombres'] + df['Mujeres'])) * 100
        etiquetas, y1, y2 = (df['Edades'], df['%Hombres'], df['%Mujeres'])
        plt.figure('Gráfica: Composición por Edades segun su Sexo', figsize=(12, 5))
        X = np.arange(16)
        pl.bar(X, +y1, color='cornflowerblue', edgecolor='white')
        pl.bar(X, -y2, color='pink', edgecolor='white')
        for x, y in zip(X, y1):
            pl.text(x + 0.1, y + 0.025, '%.2f ' % y+'%', ha='center', va='bottom', fontsize=8, color='black')
        for x, y in zip(X, y2):
            pl.text(x + 0.1, -y + 0.025, '%.2f' % y+'%', ha='center', va='top', fontsize=8, color='black')
        pl.title('Composición por Edades segun su Sexo')
        pl.legend(labels=['Hombres', 'Mujeres'])
        pl.xticks(X, ["0-04", "05-09", "10-14", "15-19", " 20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                      "50-54", "55-59", "60-64", "65-69", "70-74", "75-más"])
        pl.ylabel('Porcentaje por edades')
        pl.xlabel('Edades')
        pl.ylim(-70.0, +80.0)
        plt.show()

    def K_means_Hombres(df):
        df = df.loc[(~df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Dimension', 'Estimador', 'Poblacion Total','Mujeres'], axis=1)

        # Representación gráfica de los datos.
        x = df['Hombres'].values
        y = df['Edades'].values
        plt.figure('Gráfica: K-Means Población de Hombres conforme a las Edades', figsize=(10, 7))

        plt.xlabel('Hombres')
        plt.ylabel('Edades')
        plt.title('Población de Hombres conforme a las Edades')

        # Determinar el número óptimo de clústeres
        plt.xlabel('Número de clústeres (k)')
        plt.ylabel('Suma de los errores cuadráticos')

        # Aplicación de k-means con k = 3
        kmeans = KMeans(n_clusters=3).fit(df)
        centroids = kmeans.cluster_centers_

        # Etiquetado de datos.
        labels = kmeans.predict(df) # Asignar cada registro de nuestro dataset a uno de los clústers
        df['label'] = labels

        # Representación gráfica de los clústeres k-means.
        colores = ['red', 'green', 'blue','yellow']
        asignar = []

        for row in labels:
            asignar.append(colores[row])
        plt.scatter(x, y, c=asignar, s=5)
        plt.scatter(centroids[:, 1], centroids[:, 0], marker='*', c='black', s=20)  # Marco centroides.
        plt.yticks(y, ["0-04", "05-09", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                       "50-54", "55-59", "60-64", "65-69", "70-74", "75-más"])
        plt.xlabel('Hombres')
        plt.ylabel('Edades')
        plt.show()

    # << Metodo K-means Mujeres con Edades>>
    def K_means_Mujeres(df):
        df = df.loc[(~df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Dimension', 'Estimador', 'Poblacion Total','Hombres'], axis=1)

        # Representación gráfica de los datos.
        x = df['Mujeres'].values
        y = df['Edades'].values
        plt.figure('Gráfica: K-Means Población de Mujeres conforme a las Edades', figsize=(10, 7))
        plt.scatter(x, y, s=5)
        plt.xlabel('Mujeres')
        plt.ylabel('Edades')
        plt.title('Población de Mujeres conforme a las Edades')

        # Determinar el número óptimo de clústeres
        nc = range(1, 30)  # El número de iteraciones que queremos hacer.
        kmeans = [KMeans(n_clusters=i) for i in nc]
        score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
        plt.xlabel('Número de clústeres (k)')
        plt.ylabel('Suma de los errores cuadráticos')

        # Aplicación de k-means con k = 5
        kmeans = KMeans(n_clusters=3).fit(df)
        centroids = kmeans.cluster_centers_

        # Etiquetado de datos.
        labels = kmeans.predict(df) # Asignar cada registro de nuestro dataset a uno de los clústers
        df['label'] = labels

        # Representación gráfica de los clústeres k-means.
        colores = ['red', 'green', 'blue']
        asignar = []
        for row in labels:
            asignar.append(colores[row])
        plt.scatter(x, y, c=asignar, s=5)
        plt.scatter(centroids[:, 1], centroids[:, 0], marker='*', c='black', s=20)  # Marco centroides.
        plt.yticks(y, ["0-04", "05-09", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                       "50-54", "55-59", "60-64", "65-69", "70-74", "75-más"])
        plt.xlabel('Mujeres')
        plt.ylabel('Edades')
        plt.title('Población de Mujeres conforme a las Edades')
        plt.show()

    global nuevoDataset
    nuevoDataset = filtrar_dataset(df)
    nuevoDataset = filtrar_ciudades(nuevoDataset)

def main():
    #system('cls')
    subprocess.call('cls', shell=True)

    print('\n\t\t\t\t\t\tBlack Team: Proyecto Final de Estadística y Análisis de Datos')
    print('\n\nElija una de las siguiente opciones para mostrar\n')
    opcion = input('1- Gráfica Habitantes por Municipio\t\t\t\t\t2- Gráfica de Sexo por municipio\n3- Gráfica entre edades y población en BC\t\t\t'
            '4- Gráfica poblacional de hombres y su edad\n5- Gráfica poblacional de hombres y su edad\t\t\t6- Salir'
                   '\n\n::')
    if(opcion == '1'):
        Dataframe.plot_habitantes_municipio(nuevoDataset)
        main()
    elif(opcion =='2'):
        Dataframe.plot_sexo_municipio(nuevoDataset)
        main()
    elif (opcion == '3'):
        Dataframe.plot_edades_total(nuevoDataset)
        main()
    elif (opcion == '4'):
        Dataframe.K_means_Hombres(nuevoDataset)
        main()
    elif (opcion == '5'):
        Dataframe.K_means_Mujeres(nuevoDataset)
        main()
    elif (opcion == '6'):
        exit()
    else:
        print('Error Opción Invalida, intente de nuevo.')

if __name__ == "__main__":
    main()