import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CarPriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Price Prediction GUI")

        self.text_area = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.load_and_process_data()
        self.display_data_statistics()
        self.train_linear_regression_model()

    def load_and_process_data(self):
        self.show_correlation_button = tk.Button(self.root, text="Show Correlation Matrix", command=self.show_correlation)
        self.show_correlation_button.pack()
        
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
        self.text_area.insert(tk.END, "\n" + "=" * 20 + "\n")  # Separator for better visibility

    def train_linear_regression_model(self):
        selector = SelectKBest(f_regression, k=20)
        X_selected = selector.fit_transform(self.X, self.Y)
        reg = LinearRegression().fit(X_selected, self.Y)
        self.text_area.insert(tk.END, "Selected Features:\n")
        self.text_area.insert(tk.END, selector.get_support(indices=True))
        self.text_area.insert(tk.END, "\nBeta:\n")
        self.text_area.insert(tk.END, reg.coef_)
        self.text_area.insert(tk.END, "\nBeta0:\n")
        self.text_area.insert(tk.END, reg.intercept_)
        self.text_area.insert(tk.END, "\n" + "=" * 20 + "\n")  # Separator for better visibility

    def show_correlation(self):
        fig = plt.figure(figsize=(15, 15))
        corr = self.X.corr()
        plt.matshow(corr, fignum=fig.number)
        plt.xticks(range(len(corr.columns)), corr.columns, fontsize=10, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10, rotation=45)
        cb = plt.colorbar(orientation="horizontal", pad=0.02)
        cb.ax.tick_params(labelsize=10)
        
        figure_window = tk.Toplevel(self.root)
        figure_window.title("Correlation Matrix")
        canvas = FigureCanvasTkAgg(fig, master=figure_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceGUI(root)
    root.mainloop()