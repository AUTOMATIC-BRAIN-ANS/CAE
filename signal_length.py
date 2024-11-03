"""

Author: Damian Pietroń
Created: 2024-04-02
Last modified: 2024-04-03

Description: This script takes a path
to a folder with the data and calculates
the HRV parameters for the ECG signal with
given signal lenght.
"""

import argparse
import neurokit2 as nk 
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

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
    all_windows : list
        A list of all the windows.
    """
    all_windows = []
    window_size_in_seconds = window_size_in_minutes * 60
    window_start = df.index[0]
    window_end = window_start + pd.Timedelta(seconds=window_size_in_seconds) 
    estimated_window_length = window_size_in_seconds * sr

    while window_end < df.index[-1]:

        window = df.loc[window_start:window_end]
        if abs(len(window) - estimated_window_length) <= 100:
            first_peak = nk.ppg_findpeaks(window['Signal'], sampling_rate=sr)["PPG_Peaks"][0]
            last_peak = nk.ppg_findpeaks(window['Signal'], sampling_rate=sr)["PPG_Peaks"][-1]
            first_peak_time = window.index[first_peak]
            last_peak_time = window.index[last_peak]
            window = df.loc[first_peak_time:last_peak_time]
            all_windows.append(window)
        window_start = window_end
        window_end = window_start + pd.Timedelta(seconds=window_size_in_seconds)
    return all_windows

def signal_analize(df_list, sr, signal_type):
    """
    This function calculates the HRV parameters for signal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    sr : int
        The sampling rate of the data.
    signal_type : str
        The type of the signal.(ecg or abp)
    
    Returns
    -------
    hrv : pandas.DataFrame
    """
    global columns
    all_hrv = []
    for df in df_list:
        signal = df['Signal']
        if signal_type == "ecg":
            rPeaks = nk.ecg_findpeaks(signal, sampling_rate=sr)["ECG_R_Peaks"]
        elif signal_type == "abp":
            rPeaks = nk.ppg_findpeaks(signal, sampling_rate=sr)["PPG_Peaks"]
        else:
            continue
        
        try:
            hrv = nk.hrv(rPeaks, sampling_rate=sr)
            if hrv.shape != (1, 91):
                continue
            else:
                all_hrv.append(hrv)
        except Exception as e:
            print(f"Error calculating HRV: {e}")
            continue
    columns = all_hrv[0].keys().tolist()
    mean_hrv = np.mean(all_hrv, axis=0)
    return pd.DataFrame(mean_hrv)

def analyze_signal_data(input_file, signal_type, sampling_rate):
    """
    Main function that takes the path to the folder with the data and 
    calculates the HRV parameters for the ECG signal with given signal length.
    The function saves the results to a csv file.

    Parameters
    ----------
    input_file : str
        The path to the folder with the data.
    signal_type : str
        The type of the signal.(ecg or abp)
    sampling_rate : int
        The sampling rate of the data.

    Returns
    -------
    None
    """
    if signal_type == "ecg":
        column_sets = COLUMN_SETS_ECG
        output_file_name = "time_dependent_data_ecg_data.csv"
    elif signal_type == "abp":
        column_sets = COLUMN_SETS_ABP
        output_file_name = "time_dependent_data_abp_data.csv"
    else:
        print("Invalid signal type.")
        return
    
    path = input_file # r"C:\Users\Damian\Desktop\Long" # path to the folder with the data
    folder_path = Path(path)  # path to the folder with the data
    csv_files = list(folder_path.glob("*.csv"))  # list of all the csv files in the folder
    window_sizes = [3, 5, 10, 30] # list of window sizes in minutes
    data = [] # list to store the data for the signal

    for csv_file in tqdm(csv_files,):

        df = get_columns_from_file(column_sets, csv_file) # get the data from the csv file
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
            
            window_size_in_points = size * sampling_rate * 60

            if window_size_in_points > time_series_length:
                print(f"Skipping window size {size} minutes for {csv_file.name} due to insufficient data points.")
                continue

            windows = create_windows(df, size, sampling_rate) # create the windows

            if windows == []:
                print(f"Skipping window size {size} minutes for {csv_file.name} due to insufficient data points.")
                continue

            hrv = signal_analize(windows, sampling_rate, "abp") # calculate the HRV parameters   
                
            hrv_flat = hrv.values.flatten()

            hrv_values = [float(x) if not pd.isna(x) else np.nan for x in hrv_flat] 
            
            data.append([csv_file.name, size, *hrv_values])
            


    df = pd.DataFrame(data)
    df.columns = ["File", "Window size"] + columns # add the columns to the dataframe
    
    df.to_csv(output_file_name, index=False) # save the dataframe to a csv file
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process signal data.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('signal_type', type=str, choices=['ecg', 'abp'], help='Type of signal (ecg or abp)')
    parser.add_argument('sampling_rate', type=int, help='Sampling rate of the signal')

    args = parser.parse_args()
    
    input_file = args.input_file
    signal_type = args.signal_type
    sampling_rate = args.sampling_rate

    analyze_signal_data(input_file, signal_type, sampling_rate)
    

