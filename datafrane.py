import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import statistics as stats
from sklearn import preprocessing
from sklearn.cluster import KMeans

# noinspection SpellCheckingInspection
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
        # df['Total 2'] = df.iloc[:, 5:7].sum(axis=1)
        return df

    def filtrar_ciudades(df):
        #       Designar ciudades por su rando de habitantes
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

    #<< Agregar metodo y PLOT con porcentaje de habitantes por municipio en BC >>

    def plot_habitantes_municipio(df):
        #       Separar a columnas necesarias para el Plot
        ciudades = df.loc[(~df['Dimension'].str.contains('Total')) & (df['Edades'] == 17)]
        a = (ciudades['Poblacion Total'])
        b = (ciudades['Dimension'])
        #       Haciendo el Plot
        plt.pie(a, labels=b, autopct="%0.1f %%", startangle=140)
        plt.axis("equal")
        plt.title('Porcentaje de Habitantes por municipio en B.C.\n\n')
        plt.show()

    # << Agregar metodo y PLOT Composicion de la poblacion por municipio segun su sexo a.k.a % entre HyM por ciudad >>
    def plot_sexo_municipio(df):
        #       Separar a columnas necesarias para el Plot
        df = df.loc[(~df['Dimension'].str.contains('Total')) & (df['Edades'] == 17)]
        df = df.drop(['Estado','Edades','Estimador','Poblacion Total'], axis=1)
        df['%Hombres'] = (df['Hombres'] / (df['Hombres'] + df['Mujeres'])) * 100
        df['%Mujeres'] = (df['Mujeres'] / (df['Hombres'] + df['Mujeres'])) * 100
        etiquetas, y1, y2 = (df['Dimension'], df['%Hombres'], df['%Mujeres'])
        print(etiquetas)
        X = np.arange(5)
        pl.bar(X, +y1, color='cornflowerblue', edgecolor='white')
        pl.bar(X, -y2, color='pink', edgecolor='white')
        #       Desplega porcentajes para Hombres y Mujeres
        for x, y in zip(X, y1):
            pl.text(x + 0.1, y + 0.025, '%.2f ' % y + '%', ha='center', va='bottom')
        for x, y in zip(X, y2):
            pl.text(x + 0.1, -y + 0.025, '%.2f' % y + '%', ha='center', va='top')
        #       Opciones de desplegado y PLOT
        pl.title('Composicion por Municipio segun su Sexo')
        pl.legend(labels=['Hombres', 'Mujeres'])
        pl.xticks(np.arange(5), etiquetas)
        pl.ylabel('Porcentaje por sexo')
        pl.ylim(-60.0, +80.0)
        plt.show()

    # << Agregar metodo y PLOT Extraer la Edad mediana y Maxima de HyM por BC (si sobra time por Ciudad) >>
    def media_de_edades(df):
        #       Separar a columnas necesarias para el Plot
        df = df.loc[(df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Dimension', 'Estimador', 'Poblacion Total'], axis=1)
        etiquetas, y1, y2 = (df['Edades'], df['Hombres'], df['Mujeres'])

        #y1 = df['Hombres'].sort_values(ascending=True)
        #y1 = y1.sort_values(ascending=True)
        #print(y2)
        #print( 'mediana Superior es: \t\t', stats.median_high(y2), '\nMediana normal es: \t\t\t', stats.median(y2))
        print(df)

    # << Aplicar un metodo PLOT y determinar el porcentaje de las edades de BC>>
    def plot_edades_total(df):
        #       Separar a columnas necesarias para el Plot
        df = df.loc[(df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Estimador', 'Poblacion Total', 'Dimension'], axis=1)
        df['%Hombres'] = (df['Hombres'] / (df['Hombres'] + df['Mujeres'])) * 100
        df['%Mujeres'] = (df['Mujeres'] / (df['Hombres'] + df['Mujeres'])) * 100
        etiquetas, y1, y2 = (df['Edades'], df['%Hombres'], df['%Mujeres'])
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

    # << Metodo K-means Hombres con Edades>>
    def K_means_Hombres(df):
        df = df.loc[(~df['Dimension'].str.contains('Total')) & ~(df['Edades'] == 17) & ~(df['Edades'] == 0)]
        df = df.drop(['Estado', 'Dimension', 'Estimador', 'Poblacion Total','Mujeres'], axis=1)

        # Normalización de los datos
        """min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(df)  # Hay que convertir a DF el resultado.
        df = df.rename(columns={0: 'Edades', 1: 'Hombres'})"""
        #print(df)

        # Representación gráfica de los datos.
        x = df['Hombres'].values
        y = df['Edades'].values
        plt.scatter(x, y, s=5)
        plt.xlabel('Hombres')
        plt.ylabel('Edades')
        plt.title('Población de Hombres conforme a las Edades')
        #plt.plot(x, y, 'o', markersize=1)
        #plt.show()

        # Determinar el número óptimo de clústeres
        nc = range(1, 30)  # El número de iteraciones que queremos hacer.
        kmeans = [KMeans(n_clusters=i) for i in nc]
        score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
        plt.xlabel('Número de clústeres (k)')
        plt.ylabel('Suma de los errores cuadráticos')
        #plt.plot(nc, score)
        #plt.show()

        # Aplicación de k-means con k = 3
        kmeans = KMeans(n_clusters=3).fit(df)
        centroids = kmeans.cluster_centers_
        #print(centroids)

        # Etiquetado de datos.
        labels = kmeans.predict(df) # Asignar cada registro de nuestro dataset a uno de los clústers
        df['label'] = labels
        #print(df)

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

        # Normalización de los datos
        """min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(df)  # Hay que convertir a DF el resultado.
        df = df.rename(columns={0: 'Edades', 1: 'Mujeres'})"""
        #print(df)

        # Representación gráfica de los datos.
        x = df['Mujeres'].values
        y = df['Edades'].values
        plt.scatter(x, y, s=5)
        plt.xlabel('Mujeres')
        plt.ylabel('Edades')
        plt.title('Población de Mujeres conforme a las Edades')
        #plt.plot(x, y, 'o', markersize=1)
        #plt.show()

        # Determinar el número óptimo de clústeres
        nc = range(1, 30)  # El número de iteraciones que queremos hacer.
        kmeans = [KMeans(n_clusters=i) for i in nc]
        score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
        plt.xlabel('Número de clústeres (k)')
        plt.ylabel('Suma de los errores cuadráticos')
        #plt.plot(nc, score)
        #plt.show()

        # Aplicación de k-means con k = 5
        kmeans = KMeans(n_clusters=3).fit(df)
        centroids = kmeans.cluster_centers_
        #print(centroids)

        # Etiquetado de datos.
        labels = kmeans.predict(df) # Asignar cada registro de nuestro dataset a uno de los clústers
        df['label'] = labels
        #print(df)

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


    nuevoDataset = filtrar_dataset(df)
    nuevoDataset = filtrar_ciudades(nuevoDataset)
    #x, y1, y2 = separar_columnas(nuevoDataset)
    #plot_habitantes_municipio(nuevoDataset)
    #plot_sexo_municipio(nuevoDataset)
    #media_de_edades(nuevoDataset)
    #K_means_Hombres(nuevoDataset)
    #K_means_Mujeres(nuevoDataset)
    plot_edades_total(nuevoDataset)
















    #   ================================================== Prubas de codigo e ideas ==================================================
    #   ===============================================================================================================================
    """
    def tester(equipo):
        a,s,d = (equipo[0],equipo[1], equipo[2])
        return a,s,d

    team = ['Adrian','Gabriel','Jesus']
    a,s,d = tester(team)
    print(a,s,d)
    """

    #       ver las 3 listas de columnas
    #for values in izip(x, y1, y2):
    #    print(values)

    # test = nuevoDataset.groupby('Dimension').describe()
    # test.to_csv('purbas.csv')
    # print(nuevoDataset.groupby('Dimension').max())
    # print(nuevoDataset.chunksize())

    """def plot_sexo_municipio(df):
        df = df.loc[(~df['Dimension'].str.contains('Total')) & (df['Edades'] == 17)]
        df = df.drop(['Estado','Edades','Estimador','Poblacion Total'], axis=1)
        df['%Hombres'] = (df['Hombres'] / (df['Hombres'] + df['Mujeres'])) * 100
        df['%Mujeres'] = (df['Mujeres'] / (df['Hombres'] + df['Mujeres'])) * 100
        x, y1, y2 = (df['Dimension'], df['%Hombres'], df['%Mujeres'])
        #return x, y1, y2
        #df = pd.DataFrame( { 'Hombres':y1,'Mujeres':y2 }, index=x )
        #barras = df.plot.bar(rot=0)
        ind = np.arange(5)
        ind2 = df['Dimension']
        width = 0.50
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.78, 0.78])
        ax.bar(ind2, y1, width, color='cornflowerblue')
        ax.bar(ind2, y2, width, bottom=y1, color='pink')
        ax.set_ylabel('Porcentaje por Sexo')
        ax.set_xlabel('Municipios de Baja California')
        ax.set_title('Composicion por Municipio segun su Sexo')
        #ax.set_xticks(ind2, ('G1', 'G2', 'G3', 'G4', 'G5'))
        ax.set_yticks(np.arange(0, 110, 10))
        ax.legend(labels=['Hombres', 'Mujeres'])

        for numeros in zip(x,y1,y2):
            print(numeros)
        plt.show()"""