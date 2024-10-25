"""

Author: Damian Pietroń
Created: 2024-04-02
Last modified: 2024-04-03

Description: This script takes a path
to a folder with the data and calculates
the HRV parameters for the ECG signal with
given signal lenght.
"""


import neurokit2 as nk 
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np


CONTENT = [
    'HRV_RMSSD',
    "HRV_LFHF",
    "HRV_SampEn"
]

COLUMN_SETS_ECG = [
    ['ekg[]', 'Datetime'], 
    ['ecg', 'DateTime'],
    ["ekg[ekg]", "Datetime"],
    ['ekg[]', "Datetime"]
]
COLUMN_SETS_ABP = [
    ['DateTime', 'abp_cnap[mmHg]']
]

def handle_path(path):
    """
    This function takes a path to a folder and returns a list of all the csv files in the folder.

    Parameters
    ----------
    path : str
        The path to the folder with the data.
    
    Returns
    -------
    csv_files : list
        A list of all the csv files in the folder.
    """
    folder_path = Path(path)  # path to the folder with the data
    csv_files = list(folder_path.glob("*.csv"))  # list of all the csv files in the folder
    return csv_files, folder_path

def get_columns_from_file(column_sets,pat_to_df):

    """
    This function reads a csv file and returns the columns that are specified in the column_sets list.

    Parameters
    ----------
    column_sets : list
        A list of lists containing the names of the columns that should be in the csv file.
    pat_to_df : str
        The path to the csv file.
    Returns
    -------
    df_to_analize : pandas.DataFrame
        The dataframe containing the data.
    """

    df = pd.read_csv(pat_to_df, on_bad_lines='skip', sep=';', decimal=',', encoding='latin1')
    for column_set in column_sets:
        if all(column in df.columns for column in column_set):
            df_to_analize = pd.DataFrame()
            for column in column_set:
                if column != 'Datetime' and column != 'DateTime':
                    df_to_analize["Signal"] = df[column]
                else:
                    df_to_analize['Datetime'] = df[column]
            return df_to_analize
    return None

def prepare_csv(df_ecg):
    """
    This function prepares the csv file for the analysis.
    
    Parameters
    ----------
    df_ecg : pandas.DataFrame
        The dataframe containing the data.
    
    Returns
    -------
    df_ecg : pandas.DataFrame
        The dataframe containing the data.
    """
    df_ecg['Datetime'] = pd.to_datetime(df_ecg['Datetime'], origin='1899-12-30', unit='D')
    df_ecg = df_ecg.set_index('Datetime')
    df_ecg = df_ecg.dropna()
    
    return df_ecg

def create_windows(df, window_size_in_minutes, sr):
    """
    This function creates windows of the given size from the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    window_size_in_minutes : int
        The size of the window in minutes.
    sr : int
        The sampling rate of the data.
    
    Returns
    -------
    window : pandas.DataFrame
        The first window of the given size.
    """
    all_windows = []
    window_size_in_seconds = window_size_in_minutes * 60
    window_start = df.index[0]
    window_end = window_start + pd.Timedelta(seconds=window_size_in_seconds) 
    expected_length = int(window_size_in_seconds * sr)
    first_iteration = True

    while window_end < df.index[-1]:
        window = df.loc[window_start:window_end]
        window_values = window['Signal'].values
        window_start = window_end
        window_end = window_start + pd.Timedelta(seconds=window_size_in_seconds)

        peaks = nk.ppg_findpeaks(window_values, sampling_rate=sr)["PPG_Peaks"]
        first_peak = peaks[0]

        window_values = window_values[first_peak:]

        if window_end < df.index[-1] and first_iteration:
            return window_values

        if  (expected_length > len(window_values) and len(window_values) > expected_length - 200) or first_iteration:
            all_windows.append(window_values[:expected_length - sr])
        
    matrix = np.array(all_windows)
    mean_window = np.mean(matrix, axis=0)

    return mean_window  

def calculate_ecg(df, sr):
    """
    This function calculates the HRV parameters for the ECG signal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    sr : int
        The sampling rate of the data.
    
    Returns
    -------
    hrv : pandas.DataFrame
    """
    df_ecg = df
    rPeaks = nk.ecg_findpeaks(df_ecg, sampling_rate=sr)["ECG_R_Peaks"]
    hrv = nk.hrv(rPeaks, sampling_rate=sr)
    return hrv

def calculate_abp(df, sr):
    """
    This function calculates the HRV parameters for the ECG signal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    sr : int
        The sampling rate of the data.
    
    Returns
    -------
    hrv : pandas.DataFrame
    """
    df_ecg = df
    rPeaks = nk.ppg_findpeaks(df_ecg, sampling_rate=sr)["PPG_Peaks"]
    try:
        hrv = nk.hrv(rPeaks, sampling_rate=sr)
    except:
        return None
    return hrv

if __name__ == '__main__':
    
    path = r"C:\Users\Damian\Desktop\Long" # path to the folder with the data
    folder_path = Path(path)  # path to the folder with the data
    csv_files = list(folder_path.glob("*.csv"))  # list of all the csv files in the folder
    window_sizes = [3, 5, 10, 30] # list of window sizes in minutes
    data = [] # list to store the data

    for csv_file in tqdm(csv_files,):

        df = get_columns_from_file(COLUMN_SETS_ECG, csv_file) # get the data from the csv file
        print(f"Analysing {csv_file.name}")
        if df is None:
            print(f"Skipping {csv_file.name}")
            continue
        df = prepare_csv(df) # prepare the data for the analysis

        time_series_length = len(df)

        for size in tqdm(window_sizes, 
                         bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[31m', '\033[0m'), 
                         leave=False, 
                         desc=f"{csv_file.name}"): # iterate over the window sizes
            
            window_size_in_points = size * 200 * 60

            if window_size_in_points > time_series_length:
                print(f"Skipping window size {size} minutes for {csv_file.name} due to insufficient data points.")
                continue

            windows = create_windows(df, size, 200) # create the windows

            hrv = calculate_abp(windows, 200) # calculate the HRV parameters   
            if hrv is None:
                continue
            hrv_values = hrv.values.flatten()
            hrv_values = [float(x) if not pd.isna(x) else np.nan for x in hrv_values] 
            data.append([csv_file.name, size, *hrv_values])
            


    columns = hrv.keys().tolist()
    df = pd.DataFrame(data) # create a dataframe from the data list
    df.columns = ["File", "Window size"] + columns # add the columns to the dataframe
    df.to_csv('time_dependent_data_ecg_data.csv', index=False) # save the dataframe to a csv file
        
