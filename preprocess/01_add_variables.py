import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd

# Carga del dataset
df = pd.read_csv('Dataset/process/data_cleaned.csv')

# Borramos la columna ID porque no es necesario para los modelos
df = df.drop('id', axis=1)

# Variable: proporción pagada del monto facturado en el mes 4
df['pay_ratio_4'] = df['pay_amt4'] / (df['bill_amt4'] + 1)
df['pay_ratio_4'] = df['pay_ratio_4'].round(5)

# Variable 2: Máximo retraso en pagos en los últimos 6 meses
df['max_delay'] = df[[f'pay_{i}' for i in range(1, 7)]].max(axis=1)

# Variable 3: Número de meses en los que el cliente hizo algún pago (>0)
df['payment_consistency'] = df[[f'pay_amt{i}' for i in range(1, 7)]].gt(0).sum(axis=1)

# Se guarda el dataset
df.to_csv('Dataset/final_data/data_final.csv', index=False)

print(df.head(5))