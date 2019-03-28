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

phi = pd.concat([x,x2,ex,cosx, ones], axis=1)
phiLabels = []
for i in range(1, 22):
    phiLabels.append("phi" + str(i))
phi.columns = phiLabels

# do linear regression
model = linear_model.RidgeCV(cv=10,fit_intercept=False)
model.fit(phi,y)

y_pred = model.predict(phi)
score =  mean_squared_error(y_pred, y)**0.5
print(score)


w = pd.Series(model.coef_)

# write to csv
w.to_csv('output_test.csv', index=False, header=False)
print(w)