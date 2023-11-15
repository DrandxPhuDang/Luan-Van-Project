import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import pandas as pd
import serial
import torch
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

from Helper.Drand_started_helper import print_score
from launcher.Helper_launcher import plot_regression, plot_brix_dis, plot_spectrum, plot_mean_spec


class windows_system(tk.Tk):

    def __init__(self):
        # tao bien dung chung
        super().__init__()
        # tao windows chinh
        self.title("Training Model Mode")
        self.geometry("1360x760")
        self.resizable(width=False, height=False)
        icon = tk.PhotoImage(file=r"D:\Luan Van\launcher\images\logo.png")
        self.iconphoto(True, icon)
        self.configure(background='light gray')

        # tao frame main chinh
        self.main_frame = tk.Frame(bg='white')
        self.main_frame.pack(expand=True, fill='both', padx=5, pady=5)

        main_top = tk.Frame(self.main_frame, bg='white')
        main_top.pack(side='top', fill='both')

        # ------------------------------------------ Clock -------------------------------------------------------------
        def date_time():
            s_time = datetime.datetime.now().strftime('%H:%M:%S, %d-%m-%Y')
            s_time = f'{s_time}'
            time_lb.config(text=s_time)
            time_lb.after(100, date_time)

        time_lb = tk.Label(main_top, fg='dark blue', bg='white', font=('Bold', 10))
        time_lb.pack()
        date_time()

        # create button sign out account
        # img_sign_out = tk.PhotoImage(file=r"D:\Luan Van\launcher\images\logo_abt.png")
        btn_sign_out = tk.Label(main_top, text="Drand", font=('TVN-Qatar2022-Bold', 15),
                                bg='white', activebackground='white', fg='white', foreground='black', border=0)
        btn_sign_out.pack(side='left')

        # ------------------------------------------ Title -------------------------------------------------------------
        title_main = tk.Label(main_top, text="TRAINING MODEL REGRESSION MODE",
                              font=('TVN-Qatar2022-Bold', 20), bg='White', fg='RED')
        title_main.pack(side='left', expand=True, fill='both')

        main_mid = tk.Frame(self.main_frame, bg='light blue')
        main_mid.pack(side='top', expand=True, fill='both')

        left_main_mid = tk.Frame(main_mid, bg='blue')
        top_right_main_mid = tk.Frame(main_mid, bg='red')
        bottom_right_main_mid = tk.Frame(main_mid, bg='yellow')

        left_main_mid.place(x=0, y=0, relwidth=.2, relheight=1.0)
        top_right_main_mid.place(relx=0.2, y=0, relwidth=0.8, relheight=0.5)
        bottom_right_main_mid.place(relx=0.2, rely=0.5, relwidth=0.8, relheight=0.5)

        path_file_data = r"D:\Luan Van\Data\Final_Data\Know_measuring.csv"
        df = pd.read_csv(path_file_data)
        list_features = df.iloc[:0, 12:]
        features_all = [f'{e}' for e in list_features]
        X = df[features_all]
        y = df['Brix']

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        def show_score():
            score = print_score(y, y_pred)
            return score

        score_test = show_score()

        def show_plot_regression():
            plot_regression(y, y_pred, name_model='Linear', score_test=score_test, root=bottom_right_main_mid)

        def show_plot_brix_dis():
            plot_brix_dis(y, root=top_right_main_mid)

        read_data = pd.read_csv(path_file_data, sep=',')
        data = pd.DataFrame(read_data)
        Position1 = data[data['Position'] == 'Mid of Segments']
        Position2 = data[data['Position'] == 'Mid of 2 Segments']

        def show_plot_spec():
            plot_spectrum(Position1, Position1, start_col=13, root=top_right_main_mid)

        def show_plot_mean_spec():
            plot_mean_spec(Position1, Position2, start_col=13, root=bottom_right_main_mid)

        frame_bnt = tk.Frame(left_main_mid)
        frame_bnt.pack(side='top')
        bnt_reg = tk.Button(frame_bnt, text='Reg', width=8, command=show_plot_regression)
        bnt_reg.pack(side='left', expand=True)
        bnt_dis = tk.Button(frame_bnt, text='Dis', width=8, command=show_plot_brix_dis)
        bnt_dis.pack(side='left', expand=True)
        bnt_spec = tk.Button(frame_bnt, text='Spec', width=8, command=show_plot_spec)
        bnt_spec.pack(side='left', expand=True)
        bnt_mean_spec = tk.Button(frame_bnt, text='M_Spec', width=8, command=show_plot_mean_spec)
        bnt_mean_spec.pack(side='left', expand=True)

        self.mainloop()


# chay chuong trinh
if __name__ == "__main__":
    windows_system()
