# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:05:30 2023

@author: amaru
"""
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error


data_set = 'breast_cancer_dataset.csv'

data = pd.read_csv(data_set)


#Seleccionando las varibles

X = data['worst perimeter'].to_numpy()
Y = data['worst radius'].to_numpy()



Regresion = LinearRegression()
Regresion.fit(X.reshape(-1,1), Y) 

Regresion = LinearRegression()
Regresion.fit(X.reshape(-1,1), Y) 


m=Regresion.coef_
b=Regresion.intercept_

xp=np.linspace(min(X), max(X), num=2)
yp=m*xp+b

print("y=%f x+ %f"%(m[0],b))

plt.plot(xp,yp, "-", label="Ajuste lineal")
plt.legend()


Entrenamiento = Regresion.predict(X.reshape(-1,1))

ECM = mean_squared_error(y_true = Y, y_pred = Entrenamiento)

RECM = np.sqrt(ECM)
print('ECM = ', ECM)
print('REMC = ', RECM)

R2 = Regresion.score(X.reshape(-1,1), Y)
print('R2 = ', R2)

# Graficando los datos
plt.plot(X, Y, "o", label="Data")
plt.xlabel(u"$\ Worst\ Perimeter$")
plt.ylabel(u"$\ Worst\ Radius$" )
plt.grid(ls="dashed")
plt.legend()
plt.show()