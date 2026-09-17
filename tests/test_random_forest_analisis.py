import sys
import unittest
from pathlib import Path

import numpy as np


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUTA_PROYECTO))

from random_forest_analisis import calcular_metricas, crear_modelo, dividir_datos


class PruebasRandomForestAnalisis(unittest.TestCase):
    def setUp(self):
        generador = np.random.default_rng(7)
        self.X = generador.normal(size=(150, 4))
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0).astype(int)

    def test_configuracion(self):
        modelo = crear_modelo(6, 4, 10)
        self.assertEqual(modelo.n_estimators, 100)
        self.assertEqual(modelo.max_depth, 6)
        self.assertEqual(modelo.min_samples_leaf, 4)

    def test_division(self):
        entrenamiento, validacion, prueba = dividir_datos(self.X, self.y, semilla=42)
        self.assertEqual(len(entrenamiento), 90)
        self.assertEqual(len(validacion), 30)
        self.assertEqual(len(prueba), 30)

    def test_entrenamiento(self):
        modelo = crear_modelo(6, 4, 10)
        modelo.fit(self.X, self.y)
        exactitud = np.mean(modelo.predict(self.X) == self.y)
        self.assertGreater(exactitud, 0.90)

    def test_metricas(self):
        y_real = np.array([0, 0, 1, 1])
        y_predicha = np.array([0, 1, 0, 1])
        metricas, matriz = calcular_metricas(y_real, y_predicha)
        np.testing.assert_array_equal(matriz, np.array([[1, 1], [1, 1]]))
        self.assertEqual(metricas["exactitud"], 0.5)


if __name__ == "__main__":
    unittest.main()
