import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# read csv file with pandas
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# define variables
y_train = (df_train['y'])
x_train = df_train.drop(columns=['Id','y'])
x_test = df_test.drop(columns=['Id'])
w = pd.DataFrame([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]], columns=x_train.columns)

# do linear regression

# define objective function
def R(w):
    xTw = x_train.dot(w.T)
    y_pred = pd.Series(xTw.squeeze(), name=y_train.name)
    y_dif = y_train.head() - y_pred.head()
    y_dif *= y_dif
    return y_dif.sum()


xTw = x_train.dot(w.T)
y_pred = pd.Series(xTw.squeeze(), name=y_train.name)
Rw = R(w)
eps = 7.069977765122693e-14
RMSE = mean_squared_error(y_train, y_pred)**0.5 #Root Mean Squared Error
while RMSE > eps:
    w = w/2

# build the dataframe
df_y_pred = pd.DataFrame(y_pred, columns=['y'])
df = pd.concat([df_test['Id'],df_y_pred], axis=1)

# write to csv
#df.to_csv('output.csv', index=False)

#RMSE = mean_squared_error(y_train, y_pred)**0.5 #Root Mean Squared Error
print (df.head(), '\nRMSE =', RMSE)