from Training_model import Regression_predict
from Export_Data_CSV import Export_Data
from Spectrum_Plot import Spectrum_Plot_Data


# chay chuong trinh
if __name__ == "__main__":
    # Import Data
    Path_File_Data = fr'D:\Luan Van\Data\Final_Data\Final_All_Data.csv'

    # ----------------------------------------------- Export Data CSV ------------------------------------------------
    # path_calib = r'D:\Luan Van\Data\Calib\final_data_calibration.csv'
    # folder_sensor = r'D:\Luan Van\data_sensor\2023-10-02'
    # path_folder_save = r'D:\Luan Van\Data\Demo_Data'
    #
    # # parameter: [file_name, path_folder_sensor, path_save, path_calib_file, list_column, Cultivar_name]
    # Export_Data(file_name='New_Demo_Data_021023',
    #             path_folder_sensor=folder_sensor,
    #             path_save=path_folder_save,
    #             path_calib_file=path_calib, list_column=['Ratio', 'Acid', 'Brix', 'Date',
    #                                                      'Point', 'Position', 'Number'],
    #             Cultivar_name='Cultivar')

    # ----------------------------------------------- Predicted Program ------------------------------------------------
    # model_regression: [SVR, RF, PLS, ANN, R, L, XGB and Stacking]
    # prepro_or_none: [None, Prepro]
    # find_best_parameter: [None, Find]
    # Tải model từ file

    Regression_predict(path_file_data=Path_File_Data, test_size=0.3,
                       model_regression="SVR", prepro_data="Prepro", find_best_parameter="None")

