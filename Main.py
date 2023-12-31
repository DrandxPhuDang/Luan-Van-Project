from Training_model import Regression_predict
from Export_Data import Export_Data
from Spectrum_Plot import Spectrum_plot
from Export_Calibration import Export_calib


# Running Program
if __name__ == "__main__":
    '''Define'''
    Path_File_Data = r'D:\Luan Van\Data\Final_Data\Data B.csv'
    # Path_File_Data = r'D:\Luan Van\Data\Final_Data\Random_measuring.csv'

    # ----------------------------------------------- Export Data CSV --------------------------------------------------
    path_calib = r'D:\Luan Van\Data\Calib\final_data_calibration (25-10-2023).csv'
    folder_sensor = r'D:\Luan Van\data_sensor\2023-10-02'
    path_folder_save = r'D:\Luan Van\Data\Demo_Data'

    ''' Parameter of Export_Data: 
    [file_name, path_folder_sensor, path_save, path_calib_file, list_column, Cultivar_name]
    '''

    # Export_Data(file_name='DATA_02102023', path_folder_sensor=folder_sensor, path_save=path_folder_save,
    #             path_calib_file=path_calib, cultivar_name='Cultivar',
    #             list_column=['Ratio', 'Acid', 'Brix', 'Date', 'Point', 'Position', 'Number'])

    # ----------------------------------------------- Predicted Program ------------------------------------------------
    ''' Parameter of Regression_predict
    model_regression: [
            SVR, RF, PLS, ANN(Neural), R(Ridge),L(Lasso), XGB(XG_boost), KNN, GBR(GradientBoosting), 
            DT(Decision Tree), LR(Linear), ETR, ADB and Stacking
    ],
    prepro_or_none: [
            True, False
    ],
    find_best_parameter: [
            True, False
    ],
    kernel_pca: [
            True, False
    ],
    start_col_X: [
            Know_measuring: start=12
            Random_measuring: start=13
    ],
    Save_model: [
            True, False
    ],
    '''

    Regression_predict(path_file_data=Path_File_Data, start_col=13,  target='Brix', test_size=0.2,
                       model_regression='R', prepro_data=True,
                       find_best_parameter=False, kernel_pca=False, save_model=False)

    # ----------------------------------------------- Spectrum Plot ----------------------------------------------------
    Path_save_spectrum = r"D:\Luan Van\Data\spectrum"

    ''' Parameter of Spectrum_plot
    [path_file_data, start_col(start column of wavelength), path_save_spectrum, save_or_none]
    '''

    # Spectrum_plot(path_file_data=Path_File_Data, start_col=13, path_save_spectrum=Path_save_spectrum,
    #               save_or_none="None")

    # ----------------------------------------------- Export Calibration -----------------------------------------------

    path_folder = r'D:\Luan Van\data_sensor\Calib'
    Path_File_Data_sensor = r'D:\Luan Van\data_sensor\2023-10-26'

    # Export_calib(path_folder=path_folder, name_file='drand')
