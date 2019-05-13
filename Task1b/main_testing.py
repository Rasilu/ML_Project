import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# read csv file with pandas
df_train = pd.read_csv('train.csv')
x = df_train.loc[:,'x1':'x5']
y = df_train['y']

# expand df by calculating aditional features
# phi = pd.concat([x, x², e^(x), cos(x), 1])
x2 = x.apply(lambda z: pow(z,2))
ex = x.apply(lambda z: np.exp(z))
cosx = x.apply(lambda z: np.cos(z))
ones = pd.Series(np.ones(x['x1'].count()))

df = pd.concat([y,x,x2,ex,cosx, ones], axis=1)
labels = ['y']
for i in range(1, 22):
    labels.append("phi" + str(i))
df.columns = labels

### COPIED FORM TASK 1a ### (slighly modified to fit this task)

# create k folds (and store them in a list (folds))
k = 10                 # number of folds
folds = []             # list stores a dataframe of each fold
count = y.count()     # total number of rows
size = count // k      # size of one fold
df_copy = df.copy()
for i in range(0,k):
    folds.append(df_copy.head(size))
    df_copy = df_copy.drop(index=range(i*size,(i+1)*size))
phi = df.filter(regex='^phi')

# for each lambda do k-fold cross-validation and report RMSE
lam = range(320,340,1)
scores = pd.Series()  #pd.Series(index=lam, name=['lambda','mean'])
for l in lam:
    model = linear_model.Ridge(alpha=l) # create lasso regression model from scikit (define alpha = lambda)
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
        x_train = train.filter(regex='^phi')
        y_train = train['y']
        model.fit(x_train,y_train)
        # let model make prediction for the 'test' set
        x_test = test.filter(regex='^phi')
        y_test = test['y']
        y_pred = model.predict(x_test)
        # calculate RMSE and store it in list RMSEs
        RMSE = mean_squared_error(y_test, y_pred)**0.5
        RMSEs.at[i] = RMSE
    #report average of RMSEs and store it in scores
    scores.at[l] = (RMSEs.mean())

print (scores)
### COPIED FORM TASK 1a ###

#w = pd.Series(model.coef_)

# write to csv
#w.to_csv('output.csv', index=False, header=False)