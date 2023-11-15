import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter


def plot_regression(y_actual, y_pred, name_model, score_test, root):
    fig = plt.figure(figsize=(4, 3))
    plt.scatter(y_actual, y_pred, label='Data')
    plt.title(f'{name_model} Regression (R²={score_test[1]:.2f})')
    reg_pls = np.polyfit(y_actual, y_pred, deg=1)
    trend_pls = np.polyval(reg_pls, y_actual)
    plt.plot(y_actual, trend_pls, 'r', label='Line pred')
    plt.plot(y_actual, y_actual, color='green', linestyle='--', linewidth=1, label="Line fit")
    canvas_reg = FigureCanvasTkAgg(fig, master=root)
    canvas_reg.draw()
    canvas_reg.get_tk_widget().pack(side='left', fill='both')


def plot_brix_dis(y, root):
    fig = plt.figure(figsize=(4, 3))
    plt.hist(y, bins=20, edgecolor='black')
    plt.title('Brix Division')
    canvas_brix = FigureCanvasTkAgg(fig, master=root)
    canvas_brix.draw()
    canvas_brix.get_tk_widget().pack(side='left', fill='both')


def plot_spectrum(Position1, Position2, start_col, root):
    def plot(ax_plot, data, start, title=None):
        y = data.values[:, start - 1:]
        y = savgol_filter(y, 11, polyorder=2, deriv=0)
        row, _ = y.shape  # so hang cua bo du lieu
        str_x = data.columns.values[start - 1:]
        fmx = []
        for x in str_x:
            fmx.append(float(x))
        for i in range(0, row):
            ax_plot.plot(fmx, y[i])
        ax_plot.set_xticks(np.arange(int(min(fmx)), int(max(fmx)), 100))
        ax_plot.set_title(title)
        ax_plot.grid()

    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    fig.suptitle('Spectrum ', fontsize=10, fontweight='bold')
    plt.subplots_adjust(left=0.076, right=0.96)
    plot(ax1, Position1, start=start_col)
    ax1.text(1000, 0.3, 'Mid of Segments', fontsize=10, ha='center', va='center')
    plot(ax2, Position2, start=start_col)
    ax2.text(1000, 0.3, 'Mid of 2 Segments', fontsize=10, ha='center', va='center')
    canvas_spec = FigureCanvasTkAgg(fig, master=root)
    canvas_spec.draw()
    canvas_spec.get_tk_widget().pack(side='right', fill='both')


def plot_mean_spec(Position1, Position2, start_col, root):
    def plot_mean(data, label, start):
        y = data.values[:, start - 1:]
        y = savgol_filter(y, 11, polyorder=2, deriv=0)
        y_mean = np.mean(y, axis=0)
        str_x = data.columns.values[start - 1:]
        fmx = []
        for x in str_x:
            fmx.append(float(x))
        plt.plot(fmx, y_mean, label=label)
        ax_plot = plt.gca()

    fig = plt.figure(figsize=(7, 6))
    plt.title('Mean Spectrum ', fontsize=10, fontweight='bold')
    plot_mean(Position1, start=start_col, label='Mid of Segments')
    plot_mean(Position2, start=start_col, label='Mid of 2 Segments')
    canvas_mean_spec = FigureCanvasTkAgg(fig, master=root)
    canvas_mean_spec.draw()
    canvas_mean_spec.get_tk_widget().pack(side='right', fill='both')
