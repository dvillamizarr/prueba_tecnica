# Prueba Tecnica Mercadolibre

Este proyecto implementa un *pipeline de Machine Learning end-to-end* para predecir la *cantidad vendida de una publicación en MercadoLibre*.  

Incluye desde la ingesta de datos hasta el almacenamiento del modelo en un model store y el dataset en una base analítica.

**NOTA: Para poder ejecutar el codigo se requiere hacer referencia al archivo de datos, el cual por tamaño no puedo ser agregado al repositorio**

---

## 📌 Descripción General

El objetivo del proyecto es *estimar la demanda de productos publicados en MercadoLibre*, identificando los factores que más influyen en las ventas (ej. reputación del vendedor, precio, estado del producto).  

El repositorio contiene los archivos que cubren todo el ciclo de vida del modelo. El notebook `Prueba_MercadoLibre.ipynb` contiene el informe detallado de la solución, haciendo enfasis en las razones de cada decisión, la explicación de las metodologías utilizadas y el análisis de resultados. Distribuido de la siguiente manera:

1.  `DataAnalyzer`, sección en la cual:
     - Se realiza la lectura de datos crudos
     - Se realiza el análisis descriptivo inicial, revisando la distribucion y valores de las variables.
     - Se dentificar valores faltantes y extremos, asi como aplicar herramientas para tratar con ellos.
     - Se limpia la base datos, de acuerdo a los hallazgos anteriores.  

2. `FeatureEngineer` sección en la cual:
   - Se realiza la creación de variables derivadas.  
   - Se realiza encoding de variables categóricas.  
   - Se normalizan las variables continuas y se hace selección de atributos.  

3. `ModelPredict` sección en la cual:
   - se realiza la división de los datos en set de entrenamiento y test. 
   - Se entrenan 5 modelos supervisados diferentes (ej. Random Forest / LGBoost).
   - Se seleccionan los hyperparametros para cada modelo haciendo uso de validación cruzada y optimización bayesiana.
     
4. `Analisis de resultados`, secciíon en la cual:
   - Se validan los resultados con métricas de regresión (R², RMSE, MAE).
   - Se concluye acerca del modelo y sus posibles limitaciones.
     
5. `Insights de negocio`, sección en la cual:
   - Se realiza una evaluación de *Permutation Importances* para interpretar drivers de negocio.
   - Se concluye acerca del análisis, interpretando los resultados y proponiendo estrategias de negocio para cada caso.

6. `Estretagia de Monitoreo`, sección en la cual:
   -  Se presenta una propuesta para monitorear el modelo
---

## 🚀 Tecnologías Utilizadas

- *Python* (pandas, scikit-learn, matplotlib, optuna)  
- *Jupyter Notebooks* (EDA, feature engineering y experimentación)  

---

## 📂 Estructura del Repositorio

📂 Estructura del Repositorio

```bash
.
├── data_analyzer.py/    # Archivo que contiene la clase `DataAnalyzer` 
├── feature_engineer.py/                # Archivo que contiene la clase `FeatureEngineer` 
├── model_predict.py/              # Archivo que contiene la clase `ModelPredict`
├── Prueba_MercadoLibre.py/              # Archivo que el informe detallado
├── README.md            # Documentación del proyecto
```
---
## 📈 Resultados principales

- *Seller loyalty* (reputación del vendedor) es el driver más fuerte de ventas.  
- *Estado del producto* (nuevo vs usado) y *precio* son variables secundarias.  
- Variables de envío y localización tienen menor impacto relativo.


## 🔮 Próximos pasos

- Implementar monitoreo en producción (drift de features y performance del modelo).  
- Automatizar el pipeline con Airflow/Composer.  
- Explorar modelos más avanzados (redes neuronales).
