# Random Forest - análisis de desempeño

**Nombre:** Iris Giselle Balderas Arreola  
**Matrícula:** A01752895  
**Materia:** Inteligencia Artificial Avanzada para la Ciencia de Datos

En este proyecto se analiza el desempeño de un Random Forest con scikit-learn sobre el dataset Breast Cancer Wisconsin. Se utiliza una separación Train / Validation / Test para revisar bias, varianza y nivel de ajuste, y después se modifican parámetros para reducir el sobreajuste.

## Datos

Se trabajó con el dataset [Breast Cancer Wisconsin de Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Tiene 569 registros y 30 variables numéricas.

La separación utilizada fue:

- Train: 341 registros (60%)
- Validation: 114 registros (20%)
- Test: 114 registros (20%)

La división es estratificada y utiliza una semilla de 42.

## Modelo base

La configuración inicial conserva los parámetros de la entrega anterior:

- 100 árboles
- profundidad máxima de 8
- mínimo de 2 muestras para dividir un nodo
- mínimo de 1 muestra por hoja
- raíz cuadrada de las variables en cada nodo
- muestreo bootstrap

El modelo obtuvo 100% de exactitud en Train, 97.37% en Validation y 95.61% en Test. El diagnóstico fue bias bajo, varianza media y overfit leve.

## Ajuste

La búsqueda se realizó usando Train y Validation. Se compararon distintas profundidades y valores de muestras mínimas por hoja. El conjunto de Test se utilizó solamente al final.

La configuración seleccionada fue:

- profundidad máxima de 6
- mínimo de 4 muestras por hoja

El modelo ajustado obtuvo 98.83% de exactitud en Train, 97.37% en Validation y 96.49% en Test. El diagnóstico final fue bias bajo, varianza baja y fit.

## Ejecución

Primero se instalan las librerías:

```bash
python3 -m pip install -r requirements.txt
```

Después se corre el programa:

```bash
python3 random_forest_analisis.py
```

## Resultados

El programa genera dentro de `outputs`:

- métricas del modelo base y ajustado
- búsqueda de parámetros
- datos de entrenamiento, validación y prueba
- matrices de confusión
- comparación antes y después del ajuste
- curvas de aprendizaje
- predicciones del conjunto de prueba
- resumen del análisis

Las pruebas se ejecutan con:

```bash
python3 -m unittest discover -s tests -v
```
