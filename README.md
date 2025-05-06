# Default of credit card clients

## 📊 Descripción del Proyecto

Este repositorio contiene el desarrollo de un proyecto para la predicción de Default y Montos de Pago de Clientes de Tarjeta de Crédito
El objetivo es aplicar técnicas de **regresión y clasificación** utilizando el dataset **Default of Credit Card Clients** para:

- Predecir el monto a pagar en el próximo período (`PAY_AMT4`) – *tarea de regresión*
- Predecir el incumplimiento de pago (`default_payment_next_month`) – *tarea de clasificación*


## 🚀 Proceso seguido

1. **Análisis Exploratorio de Datos (EDA)**
   - Visualización de distribuciones y outliers
   - Análisis de correlación

2. **Limpieza de datos**
   - Modificación de los nombres de las columnas
   - Modificación de los tipos de datos
   - Revisión de valores atípicos y nulos

3. **Creación de variables**
   - `max_delay`: retraso máximo en pagos
   - `pay_ratio_4`: proporción de pago vs deuda en el mes 4
   - `average_bill_amt`: promedio de deuda en 6 meses

4. **Selección de variables**
   - Análisis de correlación

5. **Modelado**
   - Clasificación: Logistic Regression, Decision Tree, Gradient Boosting
   - Regresión: Linear Regression, Decision Tree, Gradient Boosting

6. **Evaluación**
   - Métricas: Accuracy, F1-score, RMSE, R²
   - Visualización: Matriz de confusión, dispersión de predicciones

7. **Conclusiones técnicas y de negocio**

## 📈 Resultados principales

### Clasificación (`default.payment.next.month`)
- **Mejor modelo**: Gradient Boosting  
- Buen desempeño en clase 0, limitado en clase 1 (desbalance de clases)

### Regresión (`PAY_AMT4`)
- **Mejor modelo**: Gradient Boosting  
- R²: **0.89**, RMSE significativamente menor que modelos lineales

## 🛠️ Tecnologías utilizadas

- Python (pandas, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Visualización y análisis exploratorio
- ML Supervised Learning (sklearn)

## 📚 Dataset

[Default of Credit Card Clients Dataset – UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## 📌 Consideraciones

- El dataset está desbalanceado, lo que afecta la calidad de predicción para la clase minoritaria.
- Se recomienda aplicar técnicas de balanceo o coste sensible en producción.

## 🧾 Autor

Desarrollado por  
[Juan Antonio Reyes Mendoza](https://github.com/JuanAntonioRe)
