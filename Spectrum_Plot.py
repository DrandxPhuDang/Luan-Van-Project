import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Spectrum_Plot_Data:

    def __init__(self, path_file_data, name_file, data_split, name_split_column,
                 name_column, non_prepro, path_spectrum, list_data, name_value_split, save_or_none):
        # tao bien dung chung
        super().__init__()

        def msc(input_data):
            input_data = np.array(input_data, dtype=np.float64)
            ref = []
            sampleCount = int(len(input_data))
            # mean centre correction
            for i in range(input_data.shape[0]):
                input_data[i, :] -= input_data[i, :].mean()
            data_msc = np.zeros_like(input_data)
            for i in range(input_data.shape[0]):
                for j in range(0, sampleCount, 10):
                    ref.append(np.mean(input_data[j:j + 10], axis=0))
                    fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
                    data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
            return data_msc

        def plot(ax, data, start, title):
            y = data.values[:, start - 1:]
            row, _ = y.shape  # so hang cua bo du lieu
            str_x = data.columns.values[start - 1:]
            fmx = []
            for x in str_x:
                fmx.append(float(x))
            for i in range(0, row):
                ax.plot(fmx, y[i])
            ax.set_xticks(np.arange(int(min(fmx)), int(max(fmx)), 100))
            ax.set_title(title)
            ax.grid()

        def plot_mean(data, label, start):
            y = data.values[:, start - 1:]
            y_mean = np.mean(y, axis=0)
            s_trx = data.columns.values[start - 1:]
            fmx = []
            for x in s_trx:
                fmx.append(float(x))
            plt.plot(fmx, y_mean, label=label)
            ax = plt.gca()
            ax.set_xlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
            ax.set_ylabel('Intensity(a.u)', fontsize=15, fontweight='bold')

        # input column starting waves
        df = pd.read_csv(path_file_data + fr'\{name_file}.csv', sep=',')
        df_loc = df[df[f'{name_split_column}'] == name_value_split]

        if data_split == 'Split':
            wavelengths_prepro_spectrum = df_loc.iloc[:0, 12:]
            full_waves_spectrum = [f'{e}' for e in wavelengths_prepro_spectrum]
            for check in range(0, len(list_data)):
                try:
                    list_data_value0 = df_loc[df_loc[f'{name_column}'] == list_data[check]]
                    list_data_value0 = list_data_value0.iloc[:, 12:]
                    prepro_list_data_value0 = msc(list_data_value0)
                    list_data_value0 = pd.DataFrame(list_data_value0, columns=full_waves_spectrum)
                    prepro_list_data_value0 = pd.DataFrame(prepro_list_data_value0, columns=full_waves_spectrum)
                    list_data_value0.to_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}' + '.csv',
                                            index=False,
                                            header=True)
                    prepro_list_data_value0.to_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}' + '.csv',
                                                   index=False,
                                                   header=True)
                except:
                    pass

            print(f'Successful export data non_prepro_spectrum excel to {path_spectrum}')

            # input column starting waves
            start_col = 1
            if non_prepro == 'None':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
                fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
                plt.subplots_adjust(left=0.076, right=0.96)
                fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
                fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
                plt.figure(figsize=(10, 7))
                plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

                for check in range(0, len(list_data)):
                    try:
                        if check < 2:
                            # load file
                            Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.grid(1)
                plt.legend()
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                if save_or_none == 'None':
                    pass

            if non_prepro == 'Prepro':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
                fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
                plt.subplots_adjust(left=0.076, right=0.96)
                fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
                fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
                plt.figure(figsize=(10, 7))
                plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

                for check in range(0, len(list_data)):
                    try:
                        if check < 2:
                            # load file
                            Data_plot = pd.read_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.legend()
                plt.grid(1)
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\prepro_spectrum.png')
                if save_or_none == 'None':
                    pass

        if data_split == 'None':
            wavelengths_prepro_spectrum = df.iloc[:0, 12:]
            full_waves_spectrum = [f'{e}' for e in wavelengths_prepro_spectrum]
            for check in range(0, len(list_data)):
                try:
                    list_data_value0 = df[df[f'{name_column}'] == list_data[check]]
                    list_data_value0 = list_data_value0.iloc[:, 12:]
                    prepro_list_data_value0 = msc(list_data_value0)
                    list_data_value0 = pd.DataFrame(list_data_value0, columns=full_waves_spectrum)
                    prepro_list_data_value0 = pd.DataFrame(prepro_list_data_value0, columns=full_waves_spectrum)
                    list_data_value0.to_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}' + '.csv',
                                            index=False,
                                            header=True)
                    prepro_list_data_value0.to_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}' + '.csv',
                                                   index=False,
                                                   header=True)
                except:
                    pass

            print(f'Successful export data non_prepro_spectrum excel to {path_spectrum}')

            # input column starting waves
            start_col = 1
            if non_prepro == 'None':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
                fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
                plt.subplots_adjust(left=0.076, right=0.96)
                fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
                fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
                plt.figure(figsize=(10, 7))
                plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

                for check in range(0, len(list_data)):
                    try:
                        if check < 2:
                            # load file
                            Data_plot = pd.read_csv(path_spectrum + r'\non_prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.grid(1)
                plt.legend()
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                if save_or_none == 'None':
                    pass

            if non_prepro == 'Prepro':

                fig, Ax = plt.subplots(2, figsize=(12, 7))
                fig.suptitle('Spectrum', fontsize=19, fontweight='bold')
                plt.subplots_adjust(left=0.076, right=0.96)
                fig.supxlabel('Wavelength(nm)', fontsize=15, fontweight='bold')
                fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
                plt.figure(figsize=(10, 7))
                plt.title('Mean Spectrum', fontsize=19, fontweight='bold')

                for check in range(0, len(list_data)):
                    try:
                        if check < 2:
                            # load file
                            Data_plot = pd.read_csv(path_spectrum + r'\prepro' + fr'\{list_data[check]}.csv',
                                                    sep=',')
                            # load plot data
                            plot(Ax[check], Data_plot, start=start_col, title=f'{list_data[check]}')
                            # load plot mean data
                            plot_mean(Data_plot, start=start_col, label=f'{list_data[check]}')
                    except:
                        print('Not have data', {list_data[check]})

                plt.legend()
                plt.grid(1)
                plt.show()
                if save_or_none == 'Save':
                    plt.savefig(path_spectrum + r'\spectrum' + r'\non_prepro_mean_spectrum.png')
                    plt.savefig(path_spectrum + r'\spectrum' + r'\prepro_spectrum.png')
                if save_or_none == 'None':
                    pass
