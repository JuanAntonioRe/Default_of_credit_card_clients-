import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset/final_data/data_final.csv')

# Variables seleccionadas por la correlación mostrada
features_class = [
    'max_delay', 'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
    'payment_consistency', 'limit_bal'
]
X_class = df[features_class]
y_class = df['default_payment_next_month']

# División train-test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print('Ha terminado la división del dataset')

# Se guardan los datasets
X_train_class.to_csv('Modelos/Clasificacion_train_test_data/X_train_class.csv', index=False)
X_test_class.to_csv('Modelos/Clasificacion_train_test_data/X_test_class.csv', index=False)
y_train_class.to_csv('Modelos/Clasificacion_train_test_data/y_train_class.csv', index=False)
y_test_class.to_csv('Modelos/Clasificacion_train_test_data/y_test_class.csv', index=False)

print('Se han guardado los datasets')