import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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



        self.next_button = Button(self.root, text="Next step",
                                              command=self.selection_data)
        self.next_button.pack(padx=10, pady=5)

        self.load_data()

    def load_data(self):
        motor_stock = pd.read_csv('CarPrice_Assignment.csv')
        self.text_area.insert(tk.END, "DataSet")
        self.text_area.insert(tk.END, motor_stock)
        self.Y = motor_stock['price']
        #variable dependiente porque depente de los valores de X, se quiere saber como calcular price
        self.X = motor_stock.drop(['price', 'car_ID', 'CarName'], axis=1)
        #Variable independiente, se eliminan columnas no necesarias


    def selection_data(self):
        self.text_area.delete('1.0', tk.END)
        #cambia caracteres por columnas y sus numeros
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
        self.next_button.configure(command=self.display_correlation_matrix)
    def display_correlation_matrix(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.corr = self.X.corr()
        self.text_area.insert(tk.END,  "Correlación \n")
        self.text_area.insert(tk.END, str(self.corr))


        self.show_correlation_button = Button(self.root, text="Show Correlation Matrix",
                                              command=self.plot_corr)
        self.show_correlation_button.pack(padx=10, pady=5)
        self.next_button.configure(command=self.feature_selection)
    def plot_corr(self):
        root2 = tk.Tk()
        self.root2 = root2
        self.root.title("Matplotlib Figure in Tkinter")
        f = plt.figure(figsize=(15, 15))
        plt.matshow(self.corr, fignum=f.number)
        plt.xticks(range(len(self.corr.columns)), self.corr.columns, fontsize=10, rotation=45)
        plt.yticks(range(len(self.corr.columns)), self.corr.columns, fontsize=10, rotation=45)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        canvas = FigureCanvasTkAgg(f, master=self.root2)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()
        root2.mainloop()
    #Selecciónar dependiendo de la regression deja solo las 20 columnas mejores respecto a Y
    def feature_selection(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.selector = SelectKBest(f_regression, k=20)
        self.X = self.selector.fit_transform(self.X, self.Y)
        # Selected features
        cols = self.selector.get_support(indices=True)
        self.attributes = self.attributes[cols]
        self.text_area.insert(tk.END,"\nSelected Features:")
        self.text_area.insert(tk.END,self.attributes)

        self.text_area.insert(tk.END,"\nNew X dataset:\n")
        self.text_area.insert(tk.END,self.X[:5])  # Print just  5 records

        self.text_area.insert(tk.END,"\nY:\n")
        self.text_area.insert(tk.END,self.Y[:5])
        self.next_button.configure(command=self.train_linear_regression_model)
   #Entrenamiento de la regression lineal
    def train_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nLinear Regression \n Price:")
        self.reg = LinearRegression().fit(self.X, self.Y)

        # Beta values
#gas*betaG+salario*betaSalario
        self.text_area.insert(tk.END,"\nBeta:\n")
        self.text_area.insert(tk.END,self.reg.coef_)

        self.text_area.insert(tk.END,"\nBeta0:\n")
        self.text_area.insert(tk.END,self.reg.intercept_)
        self.next_button.configure(command=self.Formula)
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
        self. next_button.destroy()



if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceGUI(root)
    root.mainloop()
