import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

np.random.seed(2161439)
tf.random.set_random_seed(105338)
# disclaimer: did not manage to get deterministic training results, even when setting a seed (still statistical consistency)

# read file with pandas
train = pd.read_hdf("train.h5", "train")
x_test = pd.read_hdf("test.h5", "test")
x_test.index.name = 'Id'

count=train['y'].count()
print('shuffling')
for i in range(10):
    train = train.sample(frac=1) #shuffle rows
val_size=5324
val = train.head(val_size)
train = train.tail(count-val_size)
training_set = train.copy()
ind = pd.Series(training_set.index.values)
x_val = val.filter(regex='^x').values
y_val = val['y'].values

#############
### MODEL ###
#############

layers = [120,500,500]
print('layers=',layers)
for i in range(len(layers)):
    layers[i] = tf.keras.layers.Dense(layers[i], activation=tf.nn.relu)
layers.append(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

model = tf.keras.Sequential(layers)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#############
### MODEL ###
#############

rounds=4
epochs=5
count=train['y'].count()
size=count//rounds
max_score = 0
best_epoch = 0
best_round = 0
for round in range(rounds):
    print('round', round+1)
    train = training_set.drop(ind.head(size))
    count = count-size
    ind = ind.tail(count)
    print(train['y'].count())
    x_train = training_set.filter(regex='^x').values
    y_train = training_set['y'].values

    for epoch in range(epochs):
        model.fit(x_train, y_train, epochs=1)
        y_pred = model.predict(x_val)
        predictions = []
        for i in range(len(y_pred)):
            predictions.append(np.argmax(y_pred[i]))
        acc = accuracy_score(y_val, predictions)
        if (acc > max_score):
            # write prediction of current best model to output
            y_pred = model.predict(x_test)
            predictions = []
            for i in range(len(y_pred)):
                predictions.append(np.argmax(y_pred[i]))
            out = pd.DataFrame(predictions, index=x_test.index, columns=['y'])
            out.to_csv('output.csv')

            max_score = acc
            best_epoch = epoch+1
            best_round = round+1

    print('round', round+1, 'summary:', 'best accuracy so far is' ,max_score,'in epoch', best_epoch, 'from round', best_round)