import numpy as np
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

        self.text_area = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.data_amount_scale = Scale(self.root, from_=1, to=100, orient="horizontal", label="Matrix Correlation Data Amount (%)")
        self.data_amount_scale.pack(padx=10, pady=5)

        self.show_correlation_button = Button(self.root, text="Show Correlation Matrix", command=self.display_correlation_matrix)
        self.show_correlation_button.pack(padx=10, pady=5)

        self.load_and_process_data()
        self.display_data_statistics()
        self.train_linear_regression_model()

    def load_and_process_data(self):
        motor_stock = pd.read_csv('CarPrice_Assignment.csv')
        Y = motor_stock['price']
        X = motor_stock.drop(['price', 'car_ID', 'CarName'], axis=1)
        one_hot_encoded_data = pd.get_dummies(X, columns=['fueltype', 'aspiration', 'doornumber', 'carbody',
                                                          'drivewheel', 'enginelocation', 'enginetype',
                                                          'cylindernumber', 'fuelsystem'])
        self.X = one_hot_encoded_data
        self.Y = Y

    def display_data_statistics(self):
        self.text_area.insert(tk.END, "Basic statistics:\n")
        self.text_area.insert(tk.END, self.X.describe().transpose())
        self.text_area.insert(tk.END, "\nBeta:\n")
        self.text_area.insert(tk.END, "Not computed yet")
        self.text_area.insert(tk.END, "\nBeta0:\n")
        self.text_area.insert(tk.END, "Not computed yet")

    def train_linear_regression_model(self):
        self.selector = SelectKBest(f_regression, k=20)
        self.X_selected = self.selector.fit_transform(self.X, self.Y)

    def display_correlation_matrix(self):
        data_amount_percentage = self.data_amount_scale.get()
        data_amount = int((data_amount_percentage / 100) * len(self.X_selected))

        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, "Correlation matrix (showing {} data points):\n".format(data_amount))
        
        corr = np.corrcoef(self.X_selected[:data_amount], rowvar=False)
        columns = self.X.columns[self.selector.get_support(indices=True)]
        self.text_area.insert(tk.END, str(columns) + "\n")
        self.text_area.insert(tk.END, str(corr))
        
        f = plt.figure(figsize=(8, 8))
        plt.matshow(corr, fignum=f.number)
        plt.xticks(range(len(columns)), columns, fontsize=10, rotation=45)
        plt.yticks(range(len(columns)), columns, fontsize=10, rotation=45)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceGUI(root)
    root.mainloop()