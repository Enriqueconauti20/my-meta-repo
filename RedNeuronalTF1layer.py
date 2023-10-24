import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np

from tensorflow.keras.layers import Activation, Dense

model=tf.keras.Sequential([tf.keras.layers.Dense \
                           (units=1, input_shape=[1])])

#use stochastic gradient descent for optimization 
#and the mean squared error loss function

model.compile(optimizer='sgd', loss='mean_squared_error')

# define some training data (xs as inputs and ys as outputs)
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#fit the model to the data
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))