import pandas as pd
import numpy as np
from copy import deepcopy
# invite mages for the machine learning ritual
from sklearn import multiclass  # Mage specialiced in divination    (based on the quality of the prediction crystal the ml ritual has created, this mage can make predictions on the classification of vectors by reading it's features)
from sklearn.svm import SVC     # Mage specialiced in transmutation (using a kernel as his arcane focus, this mage helps the multicalss mage perform the ml ritual)
from sklearn.metrics import accuracy_score # scolar in training, tasked with evaluating the success of the rital


# read csv file with pandas
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# construct relevant dataframes
np.random.seed(42)
df_train = df_train.reindex(np.random.permutation(df_train.index))
x_test = df_test.filter(regex='^x')

# create k folds for crossvalidation (and store them in a list (folds))
k = 10                 # number of folds
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

model = multiclass.OneVsRestClassifier(SVC(gamma='auto')) # choice of good crysal is very important

for fold in folds:
    # prepare the sacrificial input data
    df_train_copy = df_train.drop(index=fold['Id'])
    x_train = df_train_copy.filter(regex='^x')
    y_train = df_train_copy['y']
    x_eval = fold.filter(regex='^x')
    y_eval = fold['y']
    
    model.fit(x_train,y_train) # atune the crystal (becomes clear and spherical)
    y_pred = model.predict(x_eval) # make a prediction using the crystal ball

    # Evaluation (performed by scolar)
    acc = accuracy_score(y_eval, y_pred)
    print(acc)
    

####################
### Magic circle ###
####################

# predict and write to output (performed by the multiclass mage)
y_pred = model.predict(x_train)
#out = pd.DataFrame(y_pred, index=df_eval['Id'], columns=['y'])
#out.to_csv('output.csv')

# Evaluation (performed by scolar)
acc = accuracy_score(y_train, y_pred)
print(acc)
