#cd documents\cic\segundo semmestre\reinforcement learning\proyecto\red
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)
#Funcion de Perdida
from keras import backend as K
def R_loss(y_true, y_pred):
    discount = 0.5
    z = K.abs(y_true - y_pred)
    return (- (K.mean(K.log(z)*discount)))

#Lectura de archivos

probas = pd.read_csv('probasbitcoin.csv')
atributos = pd.read_csv('bitcointraining.csv')
atributos2 = atributos.iloc[:,0:4]

#modelo  de la red
model=Sequential()
model.add(Dense(16,input_dim=4,kernel_initializer='glorot_normal',bias_initializer='zeros',activation='softmax')) 
model.add(Dense(3,activation='softmax')) 
model.compile(loss=R_loss,optimizer='adam',metrics=['accuracy'])
#entrenamiento de la Red
model.fit(atributos2,probas,epochs=1000,batch_size=96)
classes = model.predict(atributos2, batch_size=96)
