import pandas as pd
import matplotlib.pyplot as plt

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

    def separar_columnas(df):
        #       Declaracion de nuestras columnas para el plot
        x, y1, y2 = (df['Dimension'], df['Hombres'], df['Mujeres'])
        return x, y1, y2

    #<< Agregar metodo y PLOT con porcentaje de habitantes por municipio en BC >>

    nuevoDataset = filtrar_dataset(df)
    nuevoDataset = filtrar_ciudades(nuevoDataset)
    x, y1, y2 = separar_columnas(nuevoDataset)

    ciudades = nuevoDataset.loc[(~nuevoDataset['Dimension'].str.contains('Total')) & (nuevoDataset['Edades'] == 17)]
    ciudades = ciudades.drop(['Estado', 'Edades', 'Estimador', 'Hombres', 'Mujeres'], axis=1)
    print(ciudades)

    a = (ciudades['Poblacion Total'])
    b = (ciudades['Dimension'])
    plt.pie(a, labels=b, autopct="%0.1f %%")
    plt.axis("equal")
    plt.show()
    # << Agregar metodo y PLOT Composicion de la poblacion por municipio segun su sexo a.k.a % entre HyM por ciudad >>



    # << Agregar metodo y PLOT Extraer la Edad mediana y Maxima de HyM por BC (si sobra time por Ciudad) >>

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
