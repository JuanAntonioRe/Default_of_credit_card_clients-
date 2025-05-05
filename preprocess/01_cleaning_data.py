import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as  pd

# Carga del dataset
df = pd.read_excel(io="Dataset/raw/default of credit card clients.xls", sheet_name = 'Data', header=None)

# Cambia el encabezado del dataset
df.columns = df.iloc[1]
df = df.iloc[2:]
df = df.reset_index(drop=True)

# Cambia el nombre de las columnas por un estilo snake_case
new_col_names = []
for old_name in df.columns:
    new_name = old_name.lower()
    new_name = new_name.replace(' ', '_')
    new_col_names.append(new_name)

df.columns = new_col_names

# Se cambia el nombre de la columna pay_0 por pay_1
df = df.rename(columns={'pay_0': 'pay_1'})

# Convertir cada columna de str a int64
for columna in df.columns:
    try:
        # Convertir explícitamente a int64
        df[columna] = df[columna].astype('int64')

    except Exception as e:
        print(f"No se pudo convertir la columna '{columna}': {e}")


# Corrección de valores 
df['education'] = df['education'].replace([0, 5, 6], 4)  # Agrupar valores no válidos como 'otros'
df['marriage'] = df['marriage'].replace(0, 3)  # Reemplazar 0 por 'otros'


print(df.head(6))
df.info()

# Verificar duplicados
dup_count = df.duplicated().sum()
print(f"\n Número de renglones duplicados: {dup_count}")

# Se guarda el dataset
df.to_csv('Dataset/process/data_cleaned.csv', index='False')