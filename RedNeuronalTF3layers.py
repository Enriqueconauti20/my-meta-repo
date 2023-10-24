"""
Mi Dataset
(X1,X2,X3,Y1)

Mi Red Neuronal de 3 capa (La IL no se cuenta)
IL  HL  HL  OL
O   O   O   O
O   O   O   
O   O   O   
    O   O

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

from tensorflow.keras.layers import Activation, Dense

import numpy as np

X=np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), \
           dtype=float)
y=np.array(([1],[0],[0],[0],[0],[0],[0],[1]), dtype=float)

model=tf.keras.Sequential()

model.add(Dense(4, input_dim=3, activation='relu', use_bias=True))
model.add(Dense(4, activation='relu', use_bias=True))

model.add(Dense(1, activation='sigmoid', use_bias=True))

model.compile(loss='mean_squared_error', optimizer='adam',\
               metrics=['binary_accuracy'])

print(model.get_weights())
history=model.fit(X,y,epochs=2000, validation_data=(X,y))

model.summary()

result=model.predict(X).round()
print(result)




