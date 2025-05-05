import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Verificación o creación de carpetas donde se van a guardar las gráficas
# Ruta carpeta
carpeta_detsino = "Docs/Graficas"
if not os.path.exists(carpeta_detsino):
    # Crea la carpeta
    os.makedirs(carpeta_detsino)

df = pd.read_csv('Dataset/process/data_cleaned.csv')

# Distribución de la variable objetivo (clasificación)
plt.figure(figsize=(6,6))
ax = sns.countplot(data=df, x='default_payment_next_month', palette='Set2')

# Agregar etiquetas
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(count, (p.get_x() + p.get_width() / 2., count), ha='center', va='bottom', fontsize=12, color='black')
    
plt.title("Distribución de Default", fontsize=14)
plt.xlabel("Default(0 = No, 1 = Si)")
plt.ylabel("Cantidad de clientes")
plt.tight_layout()

# Guardar la gráfica
plt.savefig(f'{carpeta_detsino}/Distribucion_default.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlaciones
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
# Guardar la gráfica
plt.savefig(f'{carpeta_detsino}/Correlacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplots por clase
plt.figure(figsize=(8,6))
sns.boxplot(x='default_payment_next_month', y='limit_bal', data=df, palette='Set3')
plt.title("Distribución del Límite de Crédito por Default")
plt.xlabel("Default (0 = No, 1 = Sí)")
plt.ylabel("Límite de Crédito")
plt.tight_layout()
# Guardar la gráfica
plt.savefig(f'{carpeta_detsino}/Boxplots_por_clase.png', dpi=300, bbox_inches='tight')
plt.show()

# Histograma de PAY_AMT4
df['pay_amt4'].hist(bins=40)
# Guardar la gráfica
plt.savefig(f'{carpeta_detsino}/Histograma.png', dpi=300, bbox_inches='tight')
plt.show()