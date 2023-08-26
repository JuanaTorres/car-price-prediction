# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

motor_stock=pd.read_csv('CarPrice_Assignment.csv')
motor_stock.head()

#Split X and Y
Y = motor_stock['price']
X = motor_stock.drop(['price','car_ID','CarName'], axis=1)#,['CarName'],['fueltype'],['aspiration'],['doornumber'],['carbody'],['drivewheel'],['enginelocation'],['enginetype'],['cylindernumber'],['fuelsystem'])

#Print X Matrix
print("\nX:\n")
X.head()
one_hot_encoded_data = pd.get_dummies(X, columns = ['fueltype', 'aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype',
                                                      'cylindernumber','fuelsystem'])
print(one_hot_encoded_data)
X=one_hot_encoded_data
attributes = X.columns
print("\nBasic statistics:\n")
print(X.describe().transpose())
import matplotlib.pyplot as plt

corr = X.corr()
print(corr)
f = plt.figure(figsize=(15, 15))
plt.matshow(corr, fignum=f.number)
plt.xticks(range(len(corr.columns)), corr.columns,fontsize=10, rotation=45);
plt.yticks(range(len(corr.columns)), corr.columns,fontsize=10, rotation=45);
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)

#Remove attributes X with low correlation respect to Y

from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression

#attributes = X.columns
selector = SelectKBest(f_regression, k=20)
X =selector.fit_transform(X, Y)
#Selected features
cols = selector.get_support(indices=True)
attributes = attributes[cols]
print("\nSelected Features:")
print(attributes)

print("\nNew X dataset:\n")
print(X[:5]) #Print just  5 records

print("\nY:\n")
print(Y[:5]) #Print just  5 records
import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, Y)

#Beta values

print("\nBeta:\n")
print(reg.coef_)

print("\nBeta0:\n")
print(reg.intercept_)