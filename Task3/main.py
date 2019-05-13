import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# disclaimer: did not manage to get deterministic training results, even when setting a seed

# read file with pandas
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

x_train = train.filter(regex='^x').values
y_train = train['y'].values

#############
### MODEL ###
#############

model = tf.keras.Sequential([
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#############
### MODEL ###
#############

model.fit(x_train, y_train, epochs=1)
predictions = model.predict(x_train)

y_pred = []
for i in range(len(predictions)):
    predictions[i] = np.argmax(predictions[i])
out = pd.DataFrame(y_pred, index=test.index, columns=['y'])
out.to_csv('output.csv')