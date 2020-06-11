import pandas as pd
class Dataframe():

    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 50)
    df = pd.read_csv('01_POBLACION_01_BC.csv', encoding="ISO-8859-1")

    #mod 123
    def filtrar_dataset(df):
        df = df.rename( columns = { 'INEGI. Tabulados de la Encuesta Intercensal 2015': 'Estado'
            ,'Unnamed: 1': 'Dimension', 'Unnamed: 2': 'Edades', 'Unnamed: 3': 'Estimador', 'Unnamed: 4': 'Poblacion Total'
            ,'Unnamed: 5':'Hombres', 'Unnamed: 6': 'Mujeres' } )
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

        #df['Total 2'] = df.iloc[:, 5:7].sum(axis=1)
        return df

    def filtrarCiudades(df):
        # Designar ciudades por su rando de habitantes
        df['Dimension'] = df['Dimension'].str.replace('Menos de 2 500 habitantes', 'Rosarito')
        df['Dimension'] = df['Dimension'].str.replace('2 500-14 999 habitantes', 'Tecate')
        df['Dimension'] = df['Dimension'].str.replace('15 000-49 999 habitantes', 'Ensenada')
        df['Dimension'] = df['Dimension'].str.replace('50 000-99 999 habitantes', 'Mexicali')
        df['Dimension'] = df['Dimension'].str.replace('100 000 y más habitantes', 'Tijuana')
        df.to_csv('poblacion_ciudades.csv', index=False)
        df = pd.read_csv('poblacion_ciudades.csv')
        return df

    nuevoDataset = filtrar_dataset(df)
    nuevoDataset = filtrarCiudades(nuevoDataset)

    #test = nuevoDataset.groupby('Dimension').describe()
    #test.to_csv('purbas.csv')
    #print('\nNuevo Dataset mostrandose :D')
    print(nuevoDataset.groupby('Dimension').max())
    #print(nuevoDataset.chunksize())
