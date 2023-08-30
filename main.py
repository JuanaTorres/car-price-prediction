from random import random, randrange

import numpy as np
import pandas as pd
from pandas_profiling.utils.cache import cache_file
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import scrolledtext, Scale, Button, ttk


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
        self.motor_stock = pd.read_csv('CarPrice_Assignment.csv')
        self.text_area.insert(tk.END, "DataSet")
        self.text_area.insert(tk.END, self.motor_stock)
        self.Y = self.motor_stock['price']
        # variable dependiente porque depente de los valores de X, se quiere saber como calcular price
        self.X = self.motor_stock.drop(['price', 'car_ID', 'CarName'], axis=1)
        # Variable independiente, se eliminan columnas no necesarias

    def selection_data(self):
        self.text_area.delete('1.0', tk.END)
        # cambia caracteres por columnas y sus numeros
        self.one_hot_encoded_data = pd.get_dummies(self.X, columns=['fueltype', 'aspiration', 'doornumber', 'carbody',
                                                                    'drivewheel', 'enginelocation', 'enginetype',
                                                                    'cylindernumber', 'fuelsystem'])

        self.X = self.one_hot_encoded_data
        self.text_area.insert(tk.END, self.X)
        self.attributes = self.X.columns
        self.Y = self.Y
        self.next_button.configure(command=self.display_data_statistics)

    def display_data_statistics(self):

        report = self.X.profile_report(sort=None, html={'style':{'fullwidth':True}})
        report_path = "motor_dataset_report.html"
        report.to_file(report_path)

        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, f"Estadísticas Avanzadas generadas en {report_path}\nBasic statistics:\n")
        self.text_area.insert(tk.END, self.X.describe().transpose())
        self.next_button.configure(command=self.display_correlation_matrix)

    def display_correlation_matrix(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.corr = self.X.corr()
        self.text_area.insert(tk.END, "Correlación \n")
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

    # Selecciónar dependiendo de la regression deja solo las 20 columnas mejores respecto a Y
    def feature_selection(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.selector = SelectKBest(f_regression, k=20)
        self.X = self.selector.fit_transform(self.X, self.Y)
        # Selected features
        cols = self.selector.get_support(indices=True)
        self.attributes = self.attributes[cols]
        self.text_area.insert(tk.END, "\nSelected Features:")
        self.text_area.insert(tk.END, self.attributes)

        self.text_area.insert(tk.END, "\nNew X dataset:\n")
        self.text_area.insert(tk.END, self.X[:5])  # Print just  5 records

        self.text_area.insert(tk.END, "\nY:\n")
        self.text_area.insert(tk.END, self.Y[:5])
        self.next_button.configure(command=self.train_linear_regression_model)

    # Entrenamiento de la regression lineal
    def train_linear_regression_model(self):
        from sklearn.linear_model import LinearRegression
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nLinear Regression \n Price:")
        self.reg = LinearRegression().fit(self.X, self.Y)

        # Beta values
        # gas*betaG+salario*betaSalario
        self.text_area.insert(tk.END, "\nBeta:\n")
        self.text_area.insert(tk.END, self.reg.coef_)

        self.text_area.insert(tk.END, "\nBeta0:\n")
        self.text_area.insert(tk.END, self.reg.intercept_)
        self.next_button.configure(command=self.Formula)

    def Formula(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nLinear Regression equation:")
        for x, y in zip(self.reg.coef_, self.attributes):
            self.text_area.insert(tk.END, x)
            self.text_area.insert(tk.END, "*")
            self.text_area.insert(tk.END, y)
            self.text_area.insert(tk.END, "+\n")

        self.text_area.insert(tk.END, self.reg.intercept_)
        self.next_button.configure(command=self.test)

    def test(self):
        inte = randrange(0, len(self.motor_stock))
        test = self.one_hot_encoded_data.iloc[inte]
        price = 0
        for x, y in zip(self.reg.coef_, self.attributes):
            price += test[y + ''] * x
        price += self.reg.intercept_
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nprecio calculado vs el real\n")
        self.text_area.insert(tk.END, price, ' VS ', self.Y.iloc[inte])
        self.text_area.insert(tk.END, ' VS ')
        self.text_area.insert(tk.END, self.Y.iloc[inte])
        self.text_area.insert(tk.END, "\nlos datos de prueba son:\n")
        self.text_area.insert(tk.END, test)
        self.next_button.configure(command=self.insertu)
        self.show_correlation_button.destroy()

    def user(self):
        price = 0
        for x, y in zip(self.reg.coef_, self.attributes):
            price+=x*self.data[y+""]
        if(price==0):
            price=" Estan los datos vacios"
        else:
            price += self.reg.intercept_
        self.text_area.delete('1.0', tk.END)
        self.text_area.update()
        self.text_area.insert(tk.END, "\nprecio calculado \n")
        self.text_area.insert(tk.END, price)

    def capture_data(self):
        self.data = {
            "symboling": self.symboling_var.get(),
            "wheelbase": self.wheelbase_var.get(),
            "carlength": self.carlength_var.get(),
            "carwidth": self.carwidth_var.get(),
            "carheight": self.carheight_var.get(),
            "curbweight": self.curbweight_var.get(),
            "enginesize": self.enginesize_var.get(),
            "boreratio": self.boreratio_var.get(),
            "stroke": self.stroke_var.get(),
            "compressionratio": self.compressionratio_var.get(),
            "horsepower": self.horsepower_var.get(),
            "peakrpm": self.peakrpm_var.get(),
            "citympg": self.citympg_var.get(),
            "highwaympg": self.highwaympg_var.get(),
            "fueltype_diesel": self.fueltype_diesel_var.get(),
            "fueltype_gas": self.fueltype_gas_var.get(),
            "aspiration_std": self.aspiration_std_var.get(),
            "aspiration_turbo": self.aspiration_turbo_var.get(),
            "doornumber_four": self.doornumber_four_var.get(),
            "doornumber_two": self.doornumber_two_var.get(),
            "carbody_convertible": self.carbody_convertible_var.get(),
            "carbody_hardtop": self.carbody_hardtop_var.get(),
            "carbody_hatchback": self.carbody_hatchback_var.get(),
            "carbody_sedan": self.carbody_sedan_var.get(),
            "carbody_wagon": self.carbody_wagon_var.get(),
            "drivewheel_4wd": self.drivewheel_4wd_var.get(),
            "drivewheel_fwd": self.drivewheel_fwd_var.get(),
            "drivewheel_rwd": self.drivewheel_rwd_var.get(),
            "enginelocation_front": self.enginelocation_front_var.get(),
            "enginelocation_rear": self.enginelocation_rear_var.get(),
            "enginetype_dohc": self.enginetype_dohc_var.get(),
            "enginetype_dohcv": self.enginetype_dohcv_var.get(),
            "enginetype_l": self.enginetype_l_var.get(),
            "enginetype_ohc": self.enginetype_ohc_var.get(),
            "enginetype_ohcf": self.enginetype_ohcf_var.get(),
            "enginetype_ohcv": self.enginetype_ohcv_var.get(),
            "enginetype_rotor": self.enginetype_rotor_var.get(),
            "cylindernumber_eight": self.cylindernumber_eight_var.get(),
            "cylindernumber_five": self.cylindernumber_five_var.get(),
            "cylindernumber_four": self.cylindernumber_four_var.get(),
            "cylindernumber_six": self.cylindernumber_six_var.get(),
            "cylindernumber_three": self.cylindernumber_three_var.get(),
            "cylindernumber_twelve": self.cylindernumber_twelve_var.get(),
            "cylindernumber_two": self.cylindernumber_two_var.get(),
            "fuelsystem_1bbl": self.fuelsystem_1bbl_var.get(),
            "fuelsystem_2bbl": self.fuelsystem_2bbl_var.get(),
            "fuelsystem_4bbl": self.fuelsystem_4bbl_var.get(),
            "fuelsystem_idi": self.fuelsystem_idi_var.get(),
            "fuelsystem_mfi": self.fuelsystem_mfi_var.get(),
            "fuelsystem_mpfi": self.fuelsystem_mpfi_var.get(),
            "fuelsystem_spdi": self.fuelsystem_spdi_var.get(),
            "fuelsystem_spfi": self.fuelsystem_spfi_var.get()
        }
        self.user()

    def insertu(self):

        # Variables para almacenar los valores capturados
        self.symboling_var = tk.DoubleVar()
        self.wheelbase_var = tk.DoubleVar()
        self.carlength_var = tk.DoubleVar()
        self.carwidth_var = tk.DoubleVar()
        self.carheight_var = tk.DoubleVar()
        self.curbweight_var = tk.DoubleVar()
        self.enginesize_var = tk.DoubleVar()
        self.boreratio_var = tk.DoubleVar()
        self.stroke_var = tk.DoubleVar()
        self.compressionratio_var = tk.DoubleVar()
        self.horsepower_var = tk.DoubleVar()
        self.peakrpm_var = tk.DoubleVar()
        self.citympg_var = tk.DoubleVar()
        self.highwaympg_var = tk.DoubleVar()
        self.fueltype_diesel_var = tk.BooleanVar()
        self.fueltype_gas_var = tk.BooleanVar()
        self.aspiration_std_var = tk.BooleanVar()
        self.aspiration_turbo_var = tk.BooleanVar()
        self.doornumber_four_var = tk.BooleanVar()
        self.doornumber_two_var = tk.BooleanVar()
        self.carbody_convertible_var = tk.BooleanVar()
        self.carbody_hardtop_var = tk.BooleanVar()
        self.carbody_hatchback_var = tk.BooleanVar()
        self.carbody_sedan_var = tk.BooleanVar()
        self.carbody_wagon_var = tk.BooleanVar()
        self.drivewheel_4wd_var = tk.BooleanVar()
        self.drivewheel_fwd_var = tk.BooleanVar()
        self.drivewheel_rwd_var = tk.BooleanVar()
        self.enginelocation_front_var = tk.BooleanVar()
        self.enginelocation_rear_var = tk.BooleanVar()
        self.enginetype_dohc_var = tk.BooleanVar()
        self.enginetype_dohcv_var = tk.BooleanVar()
        self.enginetype_l_var = tk.BooleanVar()
        self.enginetype_ohc_var = tk.BooleanVar()
        self.enginetype_ohcf_var = tk.BooleanVar()
        self.enginetype_ohcv_var = tk.BooleanVar()
        self.enginetype_rotor_var = tk.BooleanVar()
        self.cylindernumber_eight_var = tk.BooleanVar()
        self.cylindernumber_five_var = tk.BooleanVar()
        self.cylindernumber_four_var = tk.BooleanVar()
        self.cylindernumber_six_var = tk.BooleanVar()
        self.cylindernumber_three_var = tk.BooleanVar()
        self.cylindernumber_twelve_var = tk.BooleanVar()
        self.cylindernumber_two_var = tk.BooleanVar()
        self.fuelsystem_1bbl_var = tk.BooleanVar()
        self.fuelsystem_2bbl_var = tk.BooleanVar()
        self.fuelsystem_4bbl_var = tk.BooleanVar()
        self.fuelsystem_idi_var = tk.BooleanVar()
        self.fuelsystem_mfi_var = tk.BooleanVar()
        self.fuelsystem_mpfi_var = tk.BooleanVar()
        self.fuelsystem_spdi_var = tk.BooleanVar()
        self.fuelsystem_spfi_var = tk.BooleanVar()
        self_variables = [
            "symboling",
            "wheelbase",
            "carlength",
            "carwidth",
            "carheight",
            "curbweight",
            "enginesize",
            "boreratio",
            "stroke",
            "compressionratio",
            "horsepower",
            "peakrpm",
            "citympg",
            "highwaympg"
        ]
        self_variablesB = [
            "fueltype_diesel",
            "fueltype_gas",
            "aspiration_std",
            "aspiration_turbo",
            "doornumber_four",
            "doornumber_two",
            "carbody_convertible",
            "carbody_hardtop",
            "carbody_hatchback",
            "carbody_sedan",
            "carbody_wagon",
            "drivewheel_4wd",
            "drivewheel_fwd",
            "drivewheel_rwd",
            "enginelocation_front",
            "enginelocation_rear",
            "enginetype_dohc",
            "enginetype_dohcv",
            "enginetype_l",
            "enginetype_ohc",
            "enginetype_ohcf",
            "enginetype_ohcv",
            "enginetype_rotor",
            "cylindernumber_eight",
            "cylindernumber_five",
            "cylindernumber_four",
            "cylindernumber_six",
            "cylindernumber_three",
            "cylindernumber_twelve",
            "cylindernumber_two",
            "fuelsystem_1bbl",
            "fuelsystem_2bbl",
            "fuelsystem_4bbl",
            "fuelsystem_idi",
            "fuelsystem_mfi",
            "fuelsystem_mpfi",
            "fuelsystem_spdi",
            "fuelsystem_spfi"
        ]
        variables = [
            self.symboling_var,
            self.wheelbase_var,
            self.carlength_var,
            self.carwidth_var,
            self.carheight_var,
            self.curbweight_var,
            self.enginesize_var,
            self.boreratio_var,
            self.stroke_var,
            self.compressionratio_var,
            self.horsepower_var,
            self.peakrpm_var,
            self.citympg_var,
            self.highwaympg_var
        ]
        variablesB = [
            self.fueltype_diesel_var,
            self.fueltype_gas_var,
            self.aspiration_std_var,
            self.aspiration_turbo_var,
            self.doornumber_four_var,
            self.doornumber_two_var,
            self.carbody_convertible_var,
            self.carbody_hardtop_var,
            self.carbody_hatchback_var,
            self.carbody_sedan_var,
            self.carbody_wagon_var,
            self.drivewheel_4wd_var,
            self.drivewheel_fwd_var,
            self.drivewheel_rwd_var,
            self.enginelocation_front_var,
            self.enginelocation_rear_var,
            self.enginetype_dohc_var,
            self.enginetype_dohcv_var,
            self.enginetype_l_var,
            self.enginetype_ohc_var,
            self.enginetype_ohcf_var,
            self.enginetype_ohcv_var,
            self.enginetype_rotor_var,
            self.cylindernumber_eight_var,
            self.cylindernumber_five_var,
            self.cylindernumber_four_var,
            self.cylindernumber_six_var,
            self.cylindernumber_three_var,
            self.cylindernumber_twelve_var,
            self.cylindernumber_two_var,
            self.fuelsystem_1bbl_var,
            self.fuelsystem_2bbl_var,
            self.fuelsystem_4bbl_var,
            self.fuelsystem_idi_var,
            self.fuelsystem_mfi_var,
            self.fuelsystem_mpfi_var,
            self.fuelsystem_spdi_var,
            self.fuelsystem_spfi_var
        ]
        self.boolean_options = [0, 1]
        for idx, x in enumerate(self_variables):
            if idx % 3 == 0:
                frame = tk.Frame(self.root)
                frame.pack()

            label = tk.Label(frame, text=f"{x}")
            label.pack(side=tk.LEFT)
            for v, j in zip(self_variables, variables):
                if (v) == x:
                    entry = tk.Entry(frame, textvariable=j)
                    entry.pack(side=tk.LEFT)
        label = tk.Label(self.root, text="Complete, 1 = Yes or 0=No")
        label.pack(padx=10, pady=5)
        for idx, x in enumerate(self_variablesB):
            if idx % 3 == 0:
                frame = tk.Frame(self.root)
                frame.pack()

            label = tk.Label(frame, text=f"{x}")
            label.pack(side=tk.LEFT)
            for v, j in zip(self_variablesB, variablesB):
                if (v) == x:
                    combobox = ttk.Combobox(frame, values=[0, 1], textvariable=j)
                    combobox.pack(side=tk.LEFT)

        capture_button = tk.Button(self.root, text="Capturar Datos", command=self.capture_data)
        capture_button.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceGUI(root)
    root.mainloop()
