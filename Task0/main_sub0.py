import pandas as pd
from sklearn.metrics import mean_squared_error

# define which file to read depending on if we want to train or test the model
training = 0 
if training:
    file = 'train.csv'
else:
    file = 'test.csv'

# read csv file with pandas
df_train = pd.read_csv(file)

# define variables
if file == 'train.csv':
    y = (df_train['y'])
    x = df_train.drop(columns=['Id','y'])
else:
    x = df_train.drop(columns=['Id'])

y_pred = x.T.mean()

df_y_pred = pd.DataFrame(y_pred, columns=['y'])
df = pd.concat([df_train['Id'],df_y_pred], axis=1)

#write to csv
df.to_csv('output.csv', index=False)

debug = False #debug
if debug:
    y_dif = y - y_pred
    print (y.head())
    print (y_pred.head())
    print (y_dif.head())
    RMSE = mean_squared_error(y, y_pred)**0.5 #Root Mean Squared Error
    print ("RMSE = ", RMSE)
    print (df.head())
