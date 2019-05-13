import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import neural_network
from sklearn.metrics import accuracy_score


# read file with pandas
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")
print(train)

count = train['x1'].count()
Id = pd.Series(range(count), name='Id')
df_train = pd.concat([Id,train],axis=1)

count = test['x1'].count()
Id = pd.Series(range(count), name='Id')
df_test = pd.concat([Id,test],axis=1)

# construct relevant dataframes
x_test = df_test.filter(regex='^x')
np.random.seed(6727976)
df_train = df_train.reindex(np.random.permutation(df_train.index))

# create k folds for crossvalidation (and store them in a list (folds))
k = 10                 # number of folds
count = df_train['Id'].count()     # total number of rows
size = count // k      # size of one fold
folds = []             # list stores a dataframe of each fold
df_train_copy = deepcopy(df_train)
for i in range(0,k):
    fold = pd.DataFrame(df_train_copy.head(size))
    folds.append(fold)
    df_train_copy = df_train_copy.drop(index=df_train_copy['Id'].head(size))

####################
### Magic circle ###   
####################

print("start ritual")

model = neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=400) 

max_score = 0
scores = pd.Series()
for i in range(k):
    fold = folds[i]
    not_fold = df_train.drop(index=fold['Id'])
    x_train = not_fold.filter(regex='^x')
    y_train = not_fold['y']
    x_eval = fold.filter(regex='^x')
    y_eval = fold['y']

    print("tuning model")
    model.fit(x_train,y_train) # train model
    print("predict values")
    y_pred = model.predict(x_eval) # make a prediction

    # Evaluation
    acc = accuracy_score(y_eval, y_pred)
    if (acc > max_score):
        best_model = deepcopy(model)
        best_fold = deepcopy(fold)
        max_score = acc
    scores.at[i] = acc
    print(acc)

mean_score = scores.mean()
print('mean score =', mean_score) 

# choose best model
x_eval = best_fold.filter(regex='^x')
y_eval = best_fold['y']
y_pred = best_model.predict(x_eval)
acc = accuracy_score(y_eval, y_pred)
print("best_model performance =", acc)
assert(acc == scores.max())

####################
### Magic circle ###
####################

# predict and write to output
y_pred = best_model.predict(x_test)
out = pd.DataFrame(y_pred, index=df_test['Id'], columns=['y'])
out.to_csv('output.csv')
