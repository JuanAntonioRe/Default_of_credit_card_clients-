import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

X_train_class = pd.read_csv('Modelos/Clasificacion_train_test_data/X_train_class.csv')
X_test_class = pd.read_csv('Modelos/Clasificacion_train_test_data/X_test_class.csv')
y_train_class = pd.read_csv('Modelos/Clasificacion_train_test_data/y_train_class.csv')
y_test_class = pd.read_csv('Modelos/Clasificacion_train_test_data/y_test_class.csv')

# Escalado de características
scaler = StandardScaler()
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# Modelos
models_class = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5,random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

for name, model in models_class.items():
    model.fit(X_train_class_scaled, y_train_class)
    preds = model.predict(X_test_class_scaled)
    print(f"\n🔍 {name}")
    print(classification_report(y_test_class, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_class, preds))
    
# Grafica comparativa
# Resultados que ya se vieron
cm_logreg = [[4529, 144], [1026, 301]]
cm_tree = [[4444, 230], [873, 454]]
cm_gb = [[4434, 239], [852, 475]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = ['Logistic Regression', 'Decision Tree', 'Gradient Boosting']
cms = [cm_logreg, cm_tree, cm_gb]

for ax, cm, name in zip(axes, cms, model_names):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.xaxis.set_ticklabels(['No Default', 'Default'])
    ax.yaxis.set_ticklabels(['No Default', 'Default'])

plt.suptitle('Comparativa de Matrices de Confusión por Modelo', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Modelos/Graficas_resultados/Comparacion_matriz_confusion', dpi=300, bbox_inches='tight')
plt.show()