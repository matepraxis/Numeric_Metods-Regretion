# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:57:35 2023

@author: amaru
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def modelo(x, gamma, beta, alpha):
    y = alpha*x**2 + beta*x + gamma
    return y    

# Leyendo los datos
data_set = 'iris_dataset.csv'
data = pd.read_csv(data_set)


# Seleccionando las variables

X = data['petal_length'].to_numpy()
Y = data['sepal_length'].to_numpy()



popt, pcov = curve_fit(modelo, X, Y)
print("Parámetros optimizados:", popt)
print("Matriz de covarianza de los parámetros optimizados:\n", pcov)
    

gamma, beta, alpha = popt


xp2 = np.linspace(min(X), max(X), num=200)
yp2 = modelo(xp2, gamma, beta, alpha)
plt.plot(xp2, yp2, "-", label="Modelo")


# Predicción con el modelo no lineal ajustado
Entrenamiento2 = modelo(X, gamma, beta, alpha)
ECM = mean_squared_error(y_true=Y,  y_pred = Entrenamiento2.reshape(-1,1))
print("El ECM del modelo es:", ECM)


R2 = r2_score(y_true=Y, y_pred=Entrenamiento2)
print('R2 del modelo no lineal:', R2)


# Graficando los datos
plt.plot(X, Y, "o", label="Data")
plt.xlabel(u"$\ Petal\ Length$")
plt.ylabel(u"$\ Sepal\ Length$" )
plt.grid(ls="dashed")
plt.legend()
plt.show()