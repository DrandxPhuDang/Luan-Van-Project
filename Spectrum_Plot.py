import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Spectrum_plot:
    def __init__(self, path_file_data, start_col, save_or_none, path_save_spectrum):
        super().__init__()
        read_data = pd.read_csv(path_file_data, sep=',')
        df = pd.DataFrame(read_data)

        # df1 = df[df['Number'] == 1]

        Position1 = df[df['Position'] == 'A']
        Position2 = df[df['Position'] == 'B']
        Position3 = df[df['Position'] == 'C']
        Position4 = df[df['Position'] == 'D']

        def plot(ax_plot, data, start, title):
            y = data.values[:, start - 1:]
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

        def plot_mean(data, label, start):
            y = data.values[:, start - 1:]
            y_mean = np.mean(y, axis=0)
            str_x = data.columns.values[start - 1:]
            fmx = []
            for x in str_x:
                fmx.append(float(x))
            plt.plot(fmx, y_mean, label=label)
            ax_plot = plt.gca()
            ax_plot.set_xlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
            ax_plot.set_ylabel('Intensity(a.u)', fontsize=15, fontweight='bold')

        # -----------------------------------------------------------------------
        fig, ax = plt.subplots(2, 2, figsize=(14, 7))
        fig.suptitle('Spectrum ', fontsize=19, fontweight='bold')
        plt.subplots_adjust(left=0.076, right=0.96)
        plot(ax[0, 0], Position1, start=start_col, title='A')
        plot(ax[0, 1], Position2, start=start_col, title='B')
        plot(ax[1, 0], Position3, start=start_col, title='C')
        plot(ax[1, 1], Position4, start=start_col, title='D')

        fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
        fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
        if save_or_none == 'Save':
            plt.savefig(path_save_spectrum + r'\Spectrum ' + '.png')
        if save_or_none == 'None':
            pass
        # -----------------------------------------------------------------------
        plt.figure(figsize=(10, 7))
        plt.title('Mean Spectrum ', fontsize=19, fontweight='bold')
        plot_mean(Position1, start=start_col, label='A')
        plot_mean(Position2, start=start_col, label='B')
        plot_mean(Position3, start=start_col, label='C')
        plot_mean(Position4, start=start_col, label='D')

        if save_or_none == 'Save':
            plt.savefig(path_save_spectrum + r'\Mean Spectrum ' + '.png')
        if save_or_none == 'None':
            pass

        plt.legend()
        plt.grid(1)
        plt.show()
        plt.close('all')
