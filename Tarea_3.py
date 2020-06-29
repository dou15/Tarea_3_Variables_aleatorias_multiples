#!/usr/bin/python

#importando bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit 
from numpy import random

#cargando xy.csv,  eliminar primer fila, no cabeceras
datosxy=pd.read_csv('xy.csv',skiprows=1,header=None)
del datosxy[0] #elimina la columna 0

x=np.arange(5.0,16.0,1) # valores de x
y=np.arange(5.0,26.0,1) # valores y  // numpy.ndarray

# Calculando la función marginal de x

Xs=[0 for i in range(0,11)]

for i in range(0,11):
    Xs[i]=datosxy.loc[i].sum()
    
# Graficando la función marginal de x
plt.plot(x,Xs)
plt.title('Curva función marginal X')
plt.xlabel('Valores en X')
plt.ylabel('Probabilidad asociada')
plt.show()

# Calculando la función marginal de y

Ys=[0 for y in range(0,21)]

for i in range(1,21):
    Ys[i]=datosxy.loc[:,i].sum()
    
# Graficando la función marginal de Y
plt.plot(y,Ys)
plt.title('Curva función marginal X')
plt.xlabel('Valores en Y')
plt.ylabel('Probabilidad asociada')
plt.show()

### Definiendo curva de mejor ajuste para la función marginal de X

def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))

# Determinando parametros curva de mejor ajuste
param,_=curve_fit(gaussiana,x,Xs) 

mu1=param[0]
sigma1=param[1]

# Evaluando curva de mejor ajuste
curvaX=gaussiana(x,mu1,sigma1)

# Imprime curva de mejor ajuste para la función marginal de X
plt.plot(x,curvaX)
plt.title('Curva de ajuste (modelo probabilístico) para la función de densidad marginales de X')
plt.xlabel('Valores en X')
plt.ylabel('Funcion densidad marginal')
plt.show()

### Definiendo curva de mejor ajuste para la función marginal de Y

# Determinando parametros curva de mejor ajuste 
param,_=curve_fit(gaussiana,y,Ys) 

mu2=param[0]
sigma2=param[1]

# Evaluando curva de mejor ajuste
curvaY=gaussiana(y,mu2,sigma2)

# Imprime curva de mejor ajuste para la función marginal de Y
plt.plot(y,curvaY)
plt.title('Curva de ajuste (modelo probabilístico) para la función de densidad marginales de Y')
plt.xlabel('Valores en Y')
plt.ylabel('Funcion densidad marginal')
plt.show()

# Asumiendo independencia
semilla=np.linspace(0,30,99)

newcurvaX=gaussiana(semilla,mu1,sigma1)
newcurvaY=gaussiana(semilla,mu2,sigma2)
newcurvaXY=newcurvaX*newcurvaY

plt.plot(newcurvaXY)
plt.title('Función conjunta bivariada al asumir X y Y independientes')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()
#!/usr/bin/python


# para el calculo de la correlación se utiliza el archivo xyp.csv,
# debido a que los datos se encuentran de una forma facil de manipular para este calculo

# cargando xyp.csv,  eliminar primer fila, no cabeceras
# para el calculo de la correlación se utiliza el archivo xyp.csv,
# debido a que los datos se encuentran de una forma facil de manipular para este calculo
datosxyp=pd.read_csv('xyp.csv',skiprows=1,header=None)


# multiplicando X, Y, P
datosxyp[3] = datosxyp[0] * datosxyp[1] * datosxyp[2]

correlacion=datosxyp.loc[:,3].sum()

#print(datosxyp)
print('la correlacion es:')
print(correlacion)

#del datosxyp[2] #elimina la columna probabilidades
del datosxyp[3] #elimina la columna datos intermedios calculo correlacion

coeficiente_correlacion = datosxyp.corr(method='pearson')
print(coeficiente_correlacion)

covarianza = datosxyp.cov()
print(covarianza)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(datosxyp[0],datosxyp[1],datosxyp[2], 'blue')
ax.set_title('Función de densidad conjunta')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

