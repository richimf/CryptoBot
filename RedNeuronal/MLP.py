import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import etiqueta as et

np.random.seed(7)
#tratamiento de los datos
datos,probas=et.etiquetaMetodo2(dataset)
L= etiqueta(dataset1)

dataset = pd.read_csv('historialRipple.csv')
probas = pd.read_csv('probas.csv')
dataset = pd.read_csv('historialRipple.csv')
onehot = pd.read_csv(' 	one_hot_bitcoin.csv')
atritbutos = datos_entrada_bitcoin




print(L)

#create model
model.add(Dense(23,input_dim=4,kernel_initializer='random_normal'),bias_initializer='zeros',activation='softmax')) #input_dim -> capas de entrada 5 por las 5 entradas que tenemos
model.add(Dense(3,activation='softmax')) # 1 da una decision en (-1,0,1)

#perdida = - np.mean (log(probability_of_action)*discount)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

model.fit(atributos,probas,epochs= 1000,batch_size=96)

#Evaluation
#scores = model.evaluate(X Y)
