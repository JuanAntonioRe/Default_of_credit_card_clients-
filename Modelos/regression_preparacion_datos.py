import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset/final_data/data_final.csv')

# Variables seleccionadas para regresión
features_reg = [
    'bill_amt3', 'bill_amt4', 'pay_amt3', 'pay_3', 'pay_ratio_4', 'limit_bal'
]
X_reg = df[features_reg]
y_reg = df['pay_amt4']

# División train-test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print('Ha terminado la división del dataset')

# Se guardan los datasets
X_train_reg.to_csv('Modelos/Regresion_train_test_data/X_train_reg.csv', index=False)
X_test_reg.to_csv('Modelos/Regresion_train_test_data/X_test_reg.csv', index=False)
y_train_reg.to_csv('Modelos/Regresion_train_test_data/y_train_reg.csv', index=False)
y_test_reg.to_csv('Modelos/Regresion_train_test_data/y_test_reg.csv', index=False)

print('Se han guardado los datasets')