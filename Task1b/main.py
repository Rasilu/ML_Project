import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from copy import deepcopy


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

# rename columns
labels = ['y']
for i in range(1, 22):
    labels.append("phi" + str(i))
df.columns = labels

# create k folds (and store them in a list (folds))
k = 10                 # number of folds
folds = []             # list stores a dataframe of each fold
count = y.count()      # total number of rows
size = count // k      # size of one fold
df_copy = df.copy()

for i in range(0,k):
    folds.append(df_copy.head(size))
    df_copy = df_copy.drop(index=range(i*size,(i+1)*size))
phi = df.filter(regex='^phi')

# prepare loop variables
eps = 0.0000000001
l = 330
stepsize = 0.1
dif = 1
direction = 1 # 1 means we go to the right, -1 we go to the left
skrinkstep = False
meanRMSE = 100 #set to high
bestMean = 100
scores = pd.Series()
iteration = 0

## LOOP START ##
while (np.absolute(dif) > eps and iteration < 50):
    # decide direction
    iteration += 1
    prevRMSE = meanRMSE
    
    if (dif < 0):
        direction *= -1
        if (skrinkstep):
            stepsize *= 2
        skrinkstep = True
    if (skrinkstep):
        stepsize /= 2
        
    # do a step
    if (direction > 0):
        l = l + stepsize
    else:
        l = l - stepsize

    model = linear_model.Ridge(alpha=l,fit_intercept=False) # create regression model from scikit (define alpha = lambda)
    RMSEs = pd.Series()  # stores Root Mean Squared Error for the k combinations of folds for current model
    bestRMSE = 100
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

        # store best model for current alpha in temp model
        if (RMSE < bestRMSE):
            tempModel = deepcopy(model)

    # update bestModel if better model is found
    meanRMSE = RMSEs.mean()
    if (meanRMSE < bestMean):
            bestMean = meanRMSE
            bestModel = tempModel
            bestLambda = l
            #print ("NEW BEST")
    dif = prevRMSE - meanRMSE

    print ("iteration",iteration, "lambda=",l,"direction",direction,"meanRSME=", RMSE) # information about current iteration
## LOOP END ##

print ("bestLambda:", bestLambda, "\nbestRMSE:", bestMean, "\nprivateScore:", mean_squared_error(bestModel.predict(phi), y)**0.5)

w = pd.Series(bestModel.coef_)

# write to csv
w.to_csv('output.csv', index=False, header=False)