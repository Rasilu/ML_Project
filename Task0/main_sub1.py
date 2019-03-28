import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# read csv file with pandas
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# define variables
y_train = (df_train['y'])
x_train = df_train.drop(columns=['Id','y'])
x_test = df_test.drop(columns=['Id'])

# do linear regression using scikit
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)

# build the dataframe
df_y_pred = pd.DataFrame(y_pred, columns=['y'])
df = pd.concat([df_test['Id'],df_y_pred], axis=1)

# write to csv
df.to_csv('output.csv', index=False)