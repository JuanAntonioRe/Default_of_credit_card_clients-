import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


X_train_reg = pd.read_csv('Modelos/Regresion_train_test_data/X_train_reg.csv')
X_test_reg = pd.read_csv('Modelos/Regresion_train_test_data/X_test_reg.csv')
y_train_reg = pd.read_csv('Modelos/Regresion_train_test_data/y_train_reg.csv')
y_test_reg = pd.read_csv('Modelos/Regresion_train_test_data/y_test_reg.csv')

def check_data(X, name="X"):
    print(f"\n🔎 Chequeo de datos para: {name}")
    print("Infinities:", np.isinf(X).sum().sum())
    print("NaNs:", np.isnan(X).sum().sum())
    print("Máximo valor:", np.max(X))
    print("Mínimo valor:", np.min(X))
    print("Shape:", X.shape)
    
# Revisión antes del escalado
check_data(X_train_reg, "X_train_reg")
check_data(X_test_reg, "X_test_reg")

# Limpieza si es necesario
X_train_reg = X_train_reg.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_reg = X_test_reg.replace([np.inf, -np.inf], np.nan).fillna(0)

# Escalado de características
scaler = RobustScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Modelos

models_reg = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, model in models_reg.items():
    model.fit(X_train_reg_scaled, y_train_reg)
    preds = model.predict(X_test_reg_scaled)
    if name == 'Gradient Boosting':
        y_pred_gb = preds
    
    print(f"\n📊 {name}")
    print(f"RMSE: {mean_squared_error(y_test_reg, preds, squared=False):.2f}")
    print(f"R² Score: {r2_score(y_test_reg, preds):.4f}")

# Asegurando que sean vectores unidimensionales
y_test_reg = y_test_reg.squeeze()

# Graficando predicciones
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=y_pred_gb, alpha=0.5, color='royalblue')
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         '--', color='red', label='Línea Ideal')

plt.title('Predicción vs Valor Real – Gradient Boosting (PAY_AMT4)')
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Modelos/Graficas_resultados/Prediccion_valor_real', dpi=300, bbox_inches='tight')
plt.show()
