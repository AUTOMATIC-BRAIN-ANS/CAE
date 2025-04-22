"""

Author: Damian Pietroń
Created: 2024-04-02
Last modified: 2024-11-20

Description: This script reads the data from the csv files in the given folder 
and calculates the HRV parameters for the ECG signal with given signal length.
"""

import argparse
import neurokit2 as nk 
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

COLUMN_SETS_ECG = [
    'ekg[]',
    'ecg', 
    "ekg[ekg]",
    'ekg[]', 
]

COLUMN_SETS_ABP = [
    'abp_cnap[mmHg]',
    'ABP', 
    'ABP_BaroIndex',   
    'abp_cnap[mmHg]', 
    "abp_finger[abp_finger]", 
    "abp_finger[mm_Hg]", 
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

def detect_csv_format(filepath):
    """
    Detects the delimiter and decimal point format of a CSV file by reading its first line.
    Returns delimiter and decimal character.

    Parameters
    ----------
    filepath : str
        The path to the csv file.

    Returns
    -------
    delimiter : str
        The delimiter character.    
    decimal : str
        The decimal point character
    """
    with open(filepath, 'r') as file: 
        first_line = file.readline() # read the first line of the file
    
    delimiter = ';' if ';' in first_line else ',' 

    decimal = ',' if delimiter == ';' else '.' 
    
    return delimiter, decimal

def get_columns_from_file(column_sets, path_to_df):
    """
    This function reads a csv file and returns the data from the columns that are in the column_sets list.

    Parameters
    ----------
    column_sets : list
        A list of lists containing the names of the columns that should be in the csv file.
    path_to_df : str
        The path to the csv file.

    Returns
    -------
    data : pandas.Series
        A list of the data from the columns.
    """
    delimiter, decimal = detect_csv_format(path_to_df)

    try:
        df = pd.read_csv(path_to_df, delimiter=delimiter, decimal=decimal)
    except Exception as e:
        tqdm.write(f"Error reading file: {e}")
        return None
    
    columns = df.columns
    for column in columns:
        if column in column_sets:
            return df[column]

def create_windows(r_peaks, window_size_in_minutes, sr):
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
    window_size = window_size_in_minutes * 60 * sr # window size in samples
    all_windows = [] # list to store all the windows
    r_peaks = np.array(r_peaks)
    num_windows = max(r_peaks) // window_size

    for i in range(num_windows): # iterate over the windows
        all_windows.append(r_peaks[(r_peaks >= i*window_size) & (r_peaks < (i+1)*window_size)]) # add r-peaks to the window

    bias = 0 if not all_windows else all_windows[-1][-1] # calculate the bias     

    if window_size * 0.9 + bias < max(r_peaks): # our esitmated bias is due to the tolerance of 10% of the window size
        all_windows.append(r_peaks[r_peaks >= num_windows * window_size]) # add the r-peaks to the last window
        
    tqdm.write('{} windows detected: '.format(str(len(all_windows)))) # print the number of windows
    return all_windows 


def calculate_r_peaks(df, signal_type, sr):
    """
    This function calculates the R-peaks for the ECG signal.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.

    Returns
    -------
    r_peaks : pandas.Series
        The series containing the R-peaks.
    """
    if signal_type == "ecg":
        ecg_clean = nk.ecg_clean(df, sampling_rate=sr, method="elgendi2010")
        r_peaks = nk.ecg_findpeaks(ecg_clean, sampling_rate=sr)["ECG_R_Peaks"]
    elif signal_type == "abp":
        ppg_clean = nk.ppg_clean(df, sampling_rate=sr)
        r_peaks = nk.ppg_findpeaks(ppg_clean, sampling_rate=sr)["PPG_Peaks"]
    
    return r_peaks
    

def signal_analize(r_peaks_patient, sr):
    """
    This function calculates the HRV parameters for signal.
    
    Parameters
    ----------
    r_peaks_patient : List
        The dataframe containing the data.
    sr : int
        The sampling rate of the data.

    
    Returns
    -------
    hrv : pandas.DataFrame
    """
    global columns # list of the HRV parameters
    all_hrv = [] # list to store the HRV parameters for all the windows
    for r_peak_patient in r_peaks_patient: # iterate over the windows
        try: 
            hrv = nk.hrv(r_peak_patient, sampling_rate=sr) if len(r_peak_patient) > 2 else [] # calculate the HRV parameters
            if hrv.shape != (1, 91): # if the HRV parameters are not calculated correctly
                continue             # skip the window
            else:
                all_hrv.append(hrv) # add the HRV parameters to the list
        except Exception as e:
            tqdm.write(f"Error calculating HRV: {e}")
            continue
    columns = all_hrv[0].keys().tolist() # get the names of the HRV parameters
    mean_hrv = np.mean(all_hrv, axis=0) # calculate the mean HRV parameters
    return pd.DataFrame(mean_hrv) # return the mean HRV parameters




def analyze_signal_data(input_file, signal_type, sampling_rate, outfile, make_shorter):
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
    outfile : str
        The name of the output file.

    Returns
    -------
    None
    """
    
    # set the column sets and the output file name based on the signal type
    if signal_type == "ecg":
        column_sets = COLUMN_SETS_ECG
        output_file_name = outfile + "_ecg_data.csv"
    elif signal_type == "abp":
        column_sets = COLUMN_SETS_ABP
        output_file_name = outfile + "_abp_data.csv"
    else:
        tqdm.write("Invalid signal type.")
        return 
    
    path = input_file # path to the folder with the data
    folder_path = Path(path)  # path to the folder with the data
    csv_files = list(folder_path.glob("*.csv"))  # list of all the csv files in the folder
    window_sizes = [3, 5, 10, 15] # list of window sizes in minutes
    data = [] # list to store the data for the signal

    for csv_file in tqdm(csv_files): # iterate over the csv files
        
        df = get_columns_from_file(column_sets, csv_file) # get the data from the csv file

        tqdm.write(f"Analysing {csv_file.name}") 
        if df is None:
            tqdm.write(f"Skipping {csv_file.name}")
            continue

        df = df.dropna() 
        
        if make_shorter:
            # Set signal length to 5 minutes
            df = df[:sampling_rate*60*5]

        r_peaks = calculate_r_peaks(df, signal_type, sampling_rate) 

        if r_peaks is None: 
            tqdm.write(f"Skipping {csv_file.name}, no R-peaks found.")
            continue
        
        for size in tqdm(window_sizes, 
                         bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\033[31m', '\033[0m'), 
                         leave=False, 
                         desc=f"{csv_file.name}"): # iterate over the window sizes
            
            windows = create_windows(r_peaks, size, sampling_rate) # create windows

            if windows == []: # if there are no windows skip the window size
                tqdm.write(f"Skipping window size {size} minutes for {csv_file.name} due to insufficient data points.")
                continue

            hrv = signal_analize(windows, sampling_rate) # calculate the HRV parameters   
                
            hrv_flat = hrv.values.flatten() # flatten the HRV parameters

            hrv_values = [float(x) if not pd.isna(x) else np.nan for x in hrv_flat] # convert the HRV parameters to floats 

            tqdm.write(f"\033[92m HRV values for {csv_file.name} with window size {size} minutes \033[0m") 
            
            data.append([csv_file.name, size, *hrv_values]) # add the hrv data to the list, along with the file name and window size  
            
    df = pd.DataFrame(data)
    df.columns = ["File", "Window size"] + columns # add the columns to the dataframe
    
    df.to_csv(output_file_name, index=False) # save the dataframe to a csv file
    

if __name__ == '__main__':
    # Argparse is used to parse the command line arguments
    
    parser = argparse.ArgumentParser(description="Process signal data.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('signal_type', type=str, choices=['ecg', 'abp'], help='Type of signal (ecg or abp)')
    parser.add_argument('sampling_rate', type=int, help='Sampling rate of the signal')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('make_shorter', type=bool, help='Make the signal shorter')

    args = parser.parse_args()

    input_file = args.input_file
    signal_type = args.signal_type
    sampling_rate = args.sampling_rate
    output_file = args.output_file
    make_shorter = args.make_shorter

    analyze_signal_data(input_file, signal_type, sampling_rate, output_file, make_shorter)
    

