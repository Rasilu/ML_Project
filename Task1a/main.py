import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# read csv file with pandas
df = pd.read_csv('train.csv')

# prepare a list of shuffled indexes
ID = pd.Series(df['Id']).copy()
np.random.seed(seed=1337)
np.random.shuffle(ID)

# create k folds (and store them in a list (folds))
k = 10                 # number of folds
folds = []             # list stores a dataframe of each fold
count = ID.count()     # total number of rows
size = count // k      # size of one fold
for i in range(0,k):
    ind = ID.head(size)
    ID = ID.drop(index=range(i*size,(i+1)*size))
    ind = ind.sort_values()
    fold = pd.DataFrame(df[df.Id.isin(ind)])
    folds.append(fold)


# for each lambda do k-fold cross-validation and report RMSE
lam = [0.1,1,10,100,1000]
scores = pd.Series()  #pd.Series(index=lam, name=['lambda','mean'])
for l in lam:
    model = linear_model.Ridge(alpha=l) # create ridge regression model from scikit (define alpha = lambda)
    RMSEs = pd.Series()  # stores Root Mean Squared Error for the k combinations of folds for current model
    for i in range(0,k):
        # prepare train and test set
        train = pd.DataFrame()
        for j in range(0,k):
            if (i != j):
                train = pd.concat([train,folds[j]])
            else:
                test  = folds[j]

        # fit model with 'train' set
        x = train.filter(regex='^x')
        y = train['y']
        model.fit(x,y)
        # let model make prediction for the 'test' set
        x = test.filter(regex='^x')
        y = test['y']
        y_pred = model.predict(x)
        # calculate RMSE and store it in list RMSEs
        RMSE = mean_squared_error(y, y_pred)**0.5
        RMSEs.at[i] = RMSE
    #report average of RMSEs and store it in scores
    scores.at[l] = (RMSEs.mean())

print (scores)

# write to csv
scores.to_csv('output.csv', index=False, header=False)