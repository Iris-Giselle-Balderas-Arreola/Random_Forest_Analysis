"""Análisis del desempeño de Random Forest con scikit-learn."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUTA_PROYECTO = Path(__file__).resolve().parent


def cargar_datos(ruta):
    datos = pd.read_csv(ruta)
    columnas_vacias = [columna for columna in datos.columns if columna.startswith("Unnamed")]
    datos = datos.drop(columns=columnas_vacias, errors="ignore")
    y = datos["diagnosis"].map({"B": 0, "M": 1}).to_numpy()
    variables = datos.drop(columns=["id", "diagnosis"], errors="ignore")
    X = variables.to_numpy(dtype=float)
    return datos, variables.columns.tolist(), X, y


def dividir_datos(X, y, semilla=42):
    indices = np.arange(len(X))
    entrenamiento_validacion, prueba = train_test_split(
        indices,
        test_size=0.20,
        random_state=semilla,
        stratify=y,
    )
    entrenamiento, validacion = train_test_split(
        entrenamiento_validacion,
        test_size=0.25,
        random_state=semilla,
        stratify=y[entrenamiento_validacion],
    )
    return entrenamiento, validacion, prueba


def crear_modelo(profundidad=8, hojas_minimas=1, semilla=42):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=profundidad,
        min_samples_split=2,
        min_samples_leaf=hojas_minimas,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=semilla,
        n_jobs=-1,
    )


def calcular_metricas(y_real, y_predicha):
    matriz = confusion_matrix(y_real, y_predicha, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()
    especificidad = vn / (vn + fp) if vn + fp else 0.0
    metricas = {
        "exactitud": accuracy_score(y_real, y_predicha),
        "precision": precision_score(y_real, y_predicha, zero_division=0),
        "sensibilidad": recall_score(y_real, y_predicha, zero_division=0),
        "especificidad": especificidad,
        "f1": f1_score(y_real, y_predicha, zero_division=0),
    }
    return metricas, matriz


def evaluar_modelo(modelo, X, y, conjuntos):
    filas = []
    matrices = {}
    predicciones = {}

    for nombre, indices in conjuntos.items():
        y_predicha = modelo.predict(X[indices])
        metricas, matriz = calcular_metricas(y[indices], y_predicha)
        filas.append({"conjunto": nombre, **metricas})
        matrices[nombre] = matriz
        predicciones[nombre] = y_predicha

    return pd.DataFrame(filas), matrices, predicciones


def buscar_parametros(X, y, entrenamiento, validacion, semilla=42):
    resultados = []

    for profundidad in [4, 5, 6, 8]:
        for hojas_minimas in [1, 2, 4]:
            modelo = crear_modelo(profundidad, hojas_minimas, semilla)
            modelo.fit(X[entrenamiento], y[entrenamiento])

            pred_train = modelo.predict(X[entrenamiento])
            pred_validation = modelo.predict(X[validacion])
            exactitud_train = accuracy_score(y[entrenamiento], pred_train)
            exactitud_validation = accuracy_score(y[validacion], pred_validation)
            f1_validation = f1_score(y[validacion], pred_validation)

            resultados.append(
                {
                    "profundidad": profundidad,
                    "hojas_minimas": hojas_minimas,
                    "exactitud_train": exactitud_train,
                    "exactitud_validation": exactitud_validation,
                    "f1_validation": f1_validation,
                    "diferencia_train_validation": abs(exactitud_train - exactitud_validation),
                    "exactitud_oob": modelo.oob_score_,
                }
            )

    tabla = pd.DataFrame(resultados)
    tabla = tabla.sort_values(
        ["f1_validation", "exactitud_validation", "diferencia_train_validation", "profundidad"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    mejor = tabla.iloc[0]
    return tabla, int(mejor["profundidad"]), int(mejor["hojas_minimas"])


def curva_aprendizaje(X, y, entrenamiento, validacion, profundidad, hojas_minimas, semilla=42):
    porcentajes = [0.20, 0.40, 0.60, 0.80, 1.00]
    filas = []

    for porcentaje in porcentajes:
        if porcentaje < 1.0:
            subconjunto, _ = train_test_split(
                entrenamiento,
                train_size=porcentaje,
                random_state=semilla,
                stratify=y[entrenamiento],
            )
        else:
            subconjunto = entrenamiento

        modelo = crear_modelo(profundidad, hojas_minimas, semilla)
        modelo.fit(X[subconjunto], y[subconjunto])
        exactitud_train = accuracy_score(y[subconjunto], modelo.predict(X[subconjunto]))
        exactitud_validation = accuracy_score(y[validacion], modelo.predict(X[validacion]))

        filas.append(
            {
                "filas_entrenamiento": len(subconjunto),
                "exactitud_train": exactitud_train,
                "exactitud_validation": exactitud_validation,
            }
        )

    return pd.DataFrame(filas)


def diagnosticar(metricas):
    train = metricas.loc[metricas["conjunto"] == "Train", "exactitud"].iloc[0]
    validation = metricas.loc[metricas["conjunto"] == "Validation", "exactitud"].iloc[0]
    test = metricas.loc[metricas["conjunto"] == "Test", "exactitud"].iloc[0]
    diferencia = max(abs(train - validation), abs(train - test))

    if train >= 0.95:
        bias = "Bajo"
    elif train >= 0.85:
        bias = "Medio"
    else:
        bias = "Alto"

    if diferencia <= 0.03:
        varianza = "Baja"
    elif diferencia <= 0.07:
        varianza = "Media"
    else:
        varianza = "Alta"

    if train >= 0.98 and diferencia > 0.03:
        ajuste = "Overfit leve"
    elif train < 0.85:
        ajuste = "Underfit"
    else:
        ajuste = "Fit"

    return {
        "bias": bias,
        "varianza": varianza,
        "ajuste": ajuste,
        "diferencia_maxima": diferencia,
    }


def guardar_matriz(matriz, titulo, ruta):
    figura, eje = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["B", "M"],
    ).plot(ax=eje, cmap="Blues", colorbar=False)
    eje.set_title(titulo)
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Valor real")
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def guardar_comparacion(metricas_base, metricas_ajustado, ruta):
    nombres = ["Train", "Validation", "Test"]
    base = metricas_base["exactitud"].to_numpy()
    ajustado = metricas_ajustado["exactitud"].to_numpy()
    x = np.arange(len(nombres))
    ancho = 0.35

    figura, eje = plt.subplots(figsize=(7, 4))
    barras1 = eje.bar(x - ancho / 2, base, ancho, label="Modelo base")
    barras2 = eje.bar(x + ancho / 2, ajustado, ancho, label="Modelo ajustado")
    eje.set_xticks(x, nombres)
    eje.set_ylim(0.90, 1.01)
    eje.set_ylabel("Exactitud")
    eje.set_title("Comparación antes y después del ajuste")
    eje.legend()
    eje.bar_label(barras1, labels=[f"{v:.1%}" for v in base], padding=2)
    eje.bar_label(barras2, labels=[f"{v:.1%}" for v in ajustado], padding=2)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def guardar_curva(tabla, titulo, ruta):
    figura, eje = plt.subplots(figsize=(7, 4))
    eje.plot(tabla["filas_entrenamiento"], tabla["exactitud_train"], marker="o", label="Train")
    eje.plot(tabla["filas_entrenamiento"], tabla["exactitud_validation"], marker="o", label="Validation")
    eje.set_ylim(0.88, 1.01)
    eje.set_xlabel("Filas de entrenamiento")
    eje.set_ylabel("Exactitud")
    eje.set_title(titulo)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def guardar_resultados(ruta, datos, indices, metricas_base, metricas_ajustado, busqueda, curvas, modelos, predicciones):
    ruta.mkdir(parents=True, exist_ok=True)

    metricas_base.to_csv(ruta / "metricas_modelo_base.csv", index=False)
    metricas_ajustado.to_csv(ruta / "metricas_modelo_ajustado.csv", index=False)
    busqueda.to_csv(ruta / "busqueda_parametros.csv", index=False)
    curvas["base"].to_csv(ruta / "curva_aprendizaje_base.csv", index=False)
    curvas["ajustado"].to_csv(ruta / "curva_aprendizaje_ajustado.csv", index=False)

    datos.iloc[indices["Train"]].to_csv(ruta / "datos_entrenamiento.csv", index=False)
    datos.iloc[indices["Validation"]].to_csv(ruta / "datos_validacion.csv", index=False)
    datos.iloc[indices["Test"]].to_csv(ruta / "datos_prueba.csv", index=False)

    prueba = indices["Test"]
    pd.DataFrame(
        {
            "id": datos.iloc[prueba]["id"].to_numpy(),
            "real": datos.iloc[prueba]["diagnosis"].to_numpy(),
            "prediccion_base": np.where(predicciones["base"] == 1, "M", "B"),
            "prediccion_ajustada": np.where(predicciones["ajustado"] == 1, "M", "B"),
        }
    ).to_csv(ruta / "predicciones_prueba.csv", index=False)

    resumen = {
        "modelo_base": {
            "profundidad": modelos["base"].max_depth,
            "hojas_minimas": modelos["base"].min_samples_leaf,
            "exactitud_oob": modelos["base"].oob_score_,
            "diagnostico": diagnosticar(metricas_base),
        },
        "modelo_ajustado": {
            "profundidad": modelos["ajustado"].max_depth,
            "hojas_minimas": modelos["ajustado"].min_samples_leaf,
            "exactitud_oob": modelos["ajustado"].oob_score_,
            "diagnostico": diagnosticar(metricas_ajustado),
        },
    }
    with open(ruta / "resumen_analisis.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Análisis de Random Forest con scikit-learn")
    parser.add_argument("--data", type=Path, default=RUTA_PROYECTO / "data" / "breast_cancer.csv")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=RUTA_PROYECTO / "outputs")
    argumentos = parser.parse_args()

    # Carga de datos
    datos, nombres_variables, X, y = cargar_datos(argumentos.data)

    # División de datos
    entrenamiento, validacion, prueba = dividir_datos(X, y, argumentos.semilla)
    conjuntos = {
        "Train": entrenamiento,
        "Validation": validacion,
        "Test": prueba,
    }

    # Modelo base
    modelo_base = crear_modelo(8, 1, argumentos.semilla)
    modelo_base.fit(X[entrenamiento], y[entrenamiento])
    metricas_base, matrices_base, predicciones_base = evaluar_modelo(modelo_base, X, y, conjuntos)

    # Ajuste de parámetros
    busqueda, mejor_profundidad, mejores_hojas = buscar_parametros(
        X,
        y,
        entrenamiento,
        validacion,
        argumentos.semilla,
    )

    # Modelo ajustado
    modelo_ajustado = crear_modelo(mejor_profundidad, mejores_hojas, argumentos.semilla)
    modelo_ajustado.fit(X[entrenamiento], y[entrenamiento])
    metricas_ajustado, matrices_ajustado, predicciones_ajustado = evaluar_modelo(
        modelo_ajustado,
        X,
        y,
        conjuntos,
    )

    # Curvas de aprendizaje
    curva_base = curva_aprendizaje(X, y, entrenamiento, validacion, 8, 1, argumentos.semilla)
    curva_ajustado = curva_aprendizaje(
        X,
        y,
        entrenamiento,
        validacion,
        mejor_profundidad,
        mejores_hojas,
        argumentos.semilla,
    )

    # Resultados
    guardar_resultados(
        argumentos.salida,
        datos,
        conjuntos,
        metricas_base,
        metricas_ajustado,
        busqueda,
        {"base": curva_base, "ajustado": curva_ajustado},
        {"base": modelo_base, "ajustado": modelo_ajustado},
        {"base": predicciones_base["Test"], "ajustado": predicciones_ajustado["Test"]},
    )

    guardar_matriz(
        matrices_base["Test"],
        "Matriz de confusión - modelo base",
        argumentos.salida / "matriz_confusion_base.png",
    )
    guardar_matriz(
        matrices_ajustado["Test"],
        "Matriz de confusión - modelo ajustado",
        argumentos.salida / "matriz_confusion_ajustado.png",
    )
    guardar_comparacion(
        metricas_base,
        metricas_ajustado,
        argumentos.salida / "comparacion_modelos.png",
    )
    guardar_curva(
        curva_base,
        "Curva de aprendizaje - modelo base",
        argumentos.salida / "curva_aprendizaje_base.png",
    )
    guardar_curva(
        curva_ajustado,
        "Curva de aprendizaje - modelo ajustado",
        argumentos.salida / "curva_aprendizaje_ajustado.png",
    )

    print("\nANÁLISIS DE RANDOM FOREST")
    print(f"Train: {len(entrenamiento)} filas")
    print(f"Validation: {len(validacion)} filas")
    print(f"Test: {len(prueba)} filas")
    print("\nModelo base")
    print(metricas_base.round(4).to_string(index=False))
    print("\nModelo ajustado")
    print(metricas_ajustado.round(4).to_string(index=False))
    print(f"\nProfundidad seleccionada: {mejor_profundidad}")
    print(f"Muestras mínimas por hoja: {mejores_hojas}")
    print(f"\nResultados guardados en: {argumentos.salida}")


if __name__ == "__main__":
    main()
