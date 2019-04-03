import pandas as pd
import numpy as np
from copy import deepcopy
# invite mages for the machine learning ritual
from sklearn import neural_network  # very intelligent, powerful mage (works in mysterious ways)
from sklearn.metrics import accuracy_score # scolar in training, tasked with evaluating the success of the rital


# read csv file with pandas
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# construct relevant dataframes
x_test = df_test.filter(regex='^x')
np.random.seed(6727976) # seed determined with multiple dice to boost the RNGs affinity to luck
df_train = df_train.reindex(np.random.permutation(df_train.index))

# create k folds for crossvalidation (and store them in a list (folds))
k = 10                  # number of folds
count = df_train['Id'].count()     # total number of rows
size = count // k      # size of one fold
folds = []             # list stores a dataframe of each fold
df_train_copy = deepcopy(df_train) # sacrificial dataframe
for i in range(0,k):
    fold = pd.DataFrame(df_train_copy.head(size))
    folds.append(fold)
    df_train_copy = df_train_copy.drop(index=df_train_copy['Id'].head(size))

####################
### Magic circle ###    (please do not disturb mages while ritual is in progress)
####################

# choice of good crystal is very important
crystal = neural_network.MLPClassifier(max_iter=1000, hidden_layer_sizes=400) # hehe, brony

max_score = 0
scores = pd.Series()
for i in range(k):
    # prepare the sacrificial input data
    fold = folds[i]
    not_fold = df_train.drop(index=fold['Id'])
    x_train = not_fold.filter(regex='^x')
    y_train = not_fold['y']
    x_eval = fold.filter(regex='^x')
    y_eval = fold['y']

    crystal.fit(x_train,y_train) # atune the crystal
    y_pred = crystal.predict(x_eval) # make a prediction using the crystal

    # Evaluation (performed by scolar)
    acc = accuracy_score(y_eval, y_pred)
    if (acc > max_score):
        best_crystal = deepcopy(crystal)
        best_fold = deepcopy(fold)
        max_score = acc
    scores.at[i] = acc
    print(acc)

mean_score = scores.mean()
print('mean score =', mean_score) 

# scolar is tasked with checking if the best crystal has been chosen
x_eval = best_fold.filter(regex='^x')
y_eval = best_fold['y']
y_pred = best_crystal.predict(x_eval) # make a prediction using the crystal
acc = accuracy_score(y_eval, y_pred)
print("best_crystal performance =", acc)
assert(acc == scores.max())

####################
### Magic circle ###
####################

# predict and write to output (performed by the multiclass mage)
y_pred = best_crystal.predict(x_test)
out = pd.DataFrame(y_pred, index=df_test['Id'], columns=['y'])
out.to_csv('output.csv')
