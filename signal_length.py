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
import os
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

def create_windows(df, window_size_in_minutes):
    """
    This function creates windows of the given size from the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    window_size_in_minutes : int
        The size of the window in minutes.
    
    Returns
    -------
    window : pandas.DataFrame
        The first window of the given size.
    """
    window_size_in_seconds = window_size_in_minutes * 60
    window_start = df.index[0]
    window_end = window_start + pd.Timedelta(seconds=window_size_in_seconds)
    window = df.loc[window_start:window_end]
    return window  # return only the first window

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
    rmssd : float
        The RMSSD parameter.
    lf_hf : float
        The LFHF parameter.
    sampen : float
        The SampEn parameter.
    """
    df_ecg = df['Signal']
    rPeaks = nk.ecg_findpeaks(df_ecg, sampling_rate=sr)["ECG_R_Peaks"]
    hrv = nk.hrv(rPeaks, sampling_rate=sr)
    rmssd = hrv['HRV_RMSSD'].iloc[0]
    lf_hf = hrv['HRV_LFHF'].iloc[0]
    sampen = hrv['HRV_SampEn'].iloc[0]
    return rmssd, lf_hf, sampen


if __name__ == '__main__':
    
    path = os.getenv('PATH') # path to the folder with the data
    folder_path = Path(path)  # path to the folder with the data
    csv_files = list(folder_path.glob("*.csv"))  # list of all the csv files in the folder
    window_sizes = [3, 5, 10, 30] # list of window sizes in minutes
    data = [] # list to store the data

    for csv_file in tqdm(csv_files,):
        df = get_columns_from_file(COLUMN_SETS_ECG, csv_file) # get the data from the csv file
        if df is None:
            continue
        df = prepare_csv(df) # prepare the data for the analysis
        for size in tqdm(window_sizes, 
                         bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[31m', '\033[0m'), 
                         leave=False, 
                         desc=f"{csv_file.name}"): # iterate over the window sizes
            windows = create_windows(df, size) # create the windows
            rmssd, lf_hf, sampen = calculate_ecg(windows, 200) # calculate the HRV parameters   
            print(f"ECG for size {size}: RMSSD: {rmssd}, LFHF: {lf_hf}, SampEn: {sampen}") # print the results
            data.append({
                'Size': size,
                'CSV_Path': csv_file.name,
                'RMSSD': rmssd,
                'LFHF': lf_hf,
                'SampEn': sampen
            }) # append the results to the data list
    df = pd.DataFrame(data) # create a dataframe from the data list
    df.to_csv('time_dependent_data_ecg_data.csv', index=False) # save the dataframe to a csv file
        