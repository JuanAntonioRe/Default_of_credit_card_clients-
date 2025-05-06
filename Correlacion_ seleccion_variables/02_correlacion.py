import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset/final_data/data_final.csv')

# Calculamos la matriz de correlación (todas las columnas son numericas)
correlation_matrix = df.corr()

# Correlación con la variable objetivo de clasificación
correlation_with_default = correlation_matrix['default_payment_next_month'].sort_values(ascending=False)

# Mostramos las variables más correlacionadas
print("Variables más correlacionadas con default_payment_next_month:")
print(correlation_with_default.head(10))

# Visualización
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['default_payment_next_month']].sort_values(by='default_payment_next_month', ascending=False), 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlación con default_payment_next_month')
plt.tight_layout()
plt.show()

# Correlación con la variable objetivo de regresión
correlation_with_payamt4 = correlation_matrix['pay_amt4'].sort_values(ascending=False)

print("\nVariables más correlacionadas con pay_amt4:")
print(correlation_with_payamt4.head(10))