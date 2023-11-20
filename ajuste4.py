# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:30:11 2023

@author: amaru
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def modelo(x, gamma, beta, alpha):
    y = alpha * np.log(abs(x*beta)) + gamma
    return y    

# Leyendo los datos
data_set = 'breast_cancer_dataset.csv'
data = pd.read_csv(data_set)


# Seleccionando las variables

X = data['mean symmetry'].to_numpy()
Y = data['worst symmetry'].to_numpy()


popt, pcov = curve_fit(modelo, X, Y)
print("Parámetros optimizados:", popt)
print("Matriz de covarianza de los parámetros optimizados:\n", pcov)
    

gamma, beta, alpha = popt


xp2 = np.linspace(min(X), max(X), num=200)
yp2 = modelo(xp2, gamma, beta, alpha)
plt.plot(xp2, yp2, "-", label="Modelo")


# Predicción con el modelo no lineal ajustado
Entrenamiento2 = modelo(X, gamma, beta, alpha)
ECM2 = mean_squared_error(y_true=Y,  y_pred = Entrenamiento2.reshape(-1,1))
print("El ECM del modelo dos es:", ECM2)


R2 = r2_score(y_true=Y, y_pred=Entrenamiento2)
print('R2 del modelo no lineal:', R2)


# Graficando los datos
plt.plot(X, Y, "o", label="Data")
plt.xlabel(u"$\ Mean\ Symmetry$")
plt.ylabel(u"$\ Worst\ Symmetry$" )
plt.grid(ls="dashed")
plt.legend()
plt.show()
