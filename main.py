# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  #linear algebra
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
print(attributes)import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import scrolledtext, Scale, Button

class CarPriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Price Prediction GUI")

        self.text_area = scrolledtext.ScrolledText(self.root, width=150, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.data_amount_scale = Scale(self.root, from_=1, to=100, orient="horizontal", label="Matrix Correlation Data Amount (%)")
        self.data_amount_scale.pack(padx=10, pady=5)

        self.show_correlation_button = Button(self.root, text="Show Correlation Matrix", command=self.display_correlation_matrix)
        self.show_correlation_button.pack(padx=10, pady=5)

        self.next_button = Button(self.root, text="Next step",
                                              command=self.selection_data)
        self.next_button.pack(padx=10, pady=5)

        self.load_data()

    def load_data(self):
        motor_stock = pd.read_csv('CarPrice_Assignment.csv')
        self.text_area.insert(tk.END, "DataSet")
        self.text_area.insert(tk.END, motor_stock)
        self.Y = motor_stock['price']
        self.X = motor_stock.drop(['price', 'car_ID', 'CarName'], axis=1)


    def selection_data(self):
        self.text_area.delete('1.0', tk.END)
        one_hot_encoded_data = pd.get_dummies(self.X, columns=['fueltype', 'aspiration', 'doornumber', 'carbody',
                                                          'drivewheel', 'enginelocation', 'enginetype',
                                                          'cylindernumber', 'fuelsystem'])


        self.X = one_hot_encoded_data
        self.text_area.insert(tk.END, self.X)
        self.attributes = self.X.columns
        self.Y = self.Y
        self.next_button.configure(command=self.display_data_statistics)

    def display_data_statistics(self):

        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "Basic statistics:\n")
        self.text_area.insert(tk.END, self.X.describe().transpose())
        self.next_button.configure(command=self.display_correlation_matrix())

    def train_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nLinear Regression:")
        self.reg = LinearRegression().fit(self.X, self.Y)

        # Beta values

        self.text_area.insert(tk.END,"\nBeta:\n")
        self.text_area.insert(tk.END,self.reg.coef_)

        self.text_area.insert(tk.END,"\nBeta0:\n")
        self.text_area.insert(tk.END,self.reg.intercept_)
        self.next_button.configure(command=self.Formula())
    def Formula(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nLinear Regression equation:")
        for x,y in zip(self.reg.coef_,self.attributes):
            self.text_area.insert(tk.END, x)
            self.text_area.insert(tk.END, "*")
            self.text_area.insert(tk.END, y)
            self.text_area.insert(tk.END,  "+\n")

        self.text_area.insert(tk.END,self.reg.intercept_)
    def feature_selection(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.selector = SelectKBest(f_regression, k=20)
        self.X_selected = self.selector.fit_transform(self.X, self.Y)
        # Selected features
        cols = self.selector.get_support(indices=True)
        self.text_area.insert(tk.END,"\nSelected Features:")
        self.text_area.insert(tk.END,self.attributes)

        self.text_area.insert(tk.END,"\nNew X dataset:\n")
        self.text_area.insert(tk.END,self.X[:5])  # Print just  5 records

        self.text_area.insert(tk.END,"\nY:\n")
        self.text_area.insert(tk.END,self.Y[:5])
        self.next_button.configure(command=self.train_linear_regression_model())
    def display_correlation_matrix(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        corr = self.X.corr()
        self.text_area.insert(tk.END,  "Correlación \n")
        self.text_area.insert(tk.END, str(corr))

        f = plt.figure(figsize=(15, 15))
        plt.matshow(corr, fignum=f.number)
        plt.xticks(range(len(corr.columns)), corr.columns, fontsize=10, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10, rotation=45)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.show()
        self.next_button.configure(command=self.feature_selection())

if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceGUI(root)
    root.mainloop()

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