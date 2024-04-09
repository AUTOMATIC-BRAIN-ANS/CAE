"""

Author: Damian Pietroń
Created: 2024-03-18
Last modified: 2024-03-18

Description: This script takes a path
to a folder with the data and calculates
the HRV parameters for the ABP and ECG signals.
The calculated parameters are then merged into
a single dataframe and saved as a csv file.
The script also creates Bland-Altman plots for
the calculated parameters.

"""
import neurokit2 as nk
from pyhrv import frequency_domain as fd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import numpy as np
import scipy.signal as sp_sig
import scipy.fftpack as sp_fft  
from tqdm import tqdm
from datetime import datetime
# The following lists contain the HRV parameters that will be calculated for the ABP and ECG signals.
CONTENT = [
    'HRV_SDNN',
    'HRV_RMSSD',
    'HRV_MeanNN',
    'HRV_pNN50',
    'HRV_pNN20',
    'HRV_ApEn',
    'HRV_SampEn',
    'HRV_FuzzyEn',
    'HRV_MSEn',
    'HRV_ShanEn'
]

FREQ_CONTENT = [
    'HRV_LF',
    'HRV_HF',
    'HRV_TP'
]

COLUMN_SETS_ECG = [
    ['ekg[]', 'co2[co2]', 'etco2[mmHg]'], 
    ['ecg', 'co2[mmHg]', 'etco2[mmHg]'],
    ["ekg[ekg]", "co2[co2]", "etco2[etco2]"],
    ['ekg[]', "co2[mm_Hg]", "etco2[mm_Hg]"]
    ]

COLUMN_SETS_ABP = [
    ['abp_finger[abp_finger]', "co2[mmHg]", "etco2[mmHg]"], 
    ['abp_cnap[mmHg]', "co2[mmHg]", "etco2[mmHg]"], 
    ["abp_finger[abp_finger]", "co2[co2]", "etco2[etco2]"],
    ["abp_finger[mm_Hg]", "co2[mm_Hg]", "etco2[mm_Hg]"]
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

def get_columns_from_file(filename, column_sets, sep=',', decimal='.'):
    """
    This function reads a csv file and returns the columns that are specified in the column_sets list.

    Parameters
    ----------
    filename : str
        The name of the csv file.
    column_sets : list
        A list of lists with the column names that are searched for in the csv file.
    sep : str, optional
        The separator used in the csv file. By default ','.
    decimal : str, optional
        The decimal separator used in the csv file. By default '.'.
    
    Returns
    -------
    list
        A list of the columns that are specified in the column_sets list.
    """
    df = pd.read_csv(filename, sep=sep, decimal=decimal, on_bad_lines='skip')
    for column_set in column_sets:
        if all(column in df.columns for column in column_set):
            return [df[column] for column in column_set]
    return None


def abp_calculation(csv_files, sr, dir_to_write_log):
    """
    This function takes a list of csv files and returns a dataframe with the calculated HRV parameters for the ABP signal.
    
    Parameters
    ----------
    csv_files : list
        A list of all the csv files in the folder.
    
    Returns
    -------
    final_df_abp : DataFrame
        A dataframe with the calculated HRV parameters for the ABP signal.
    """
    final_df_abp = pd.DataFrame()
    for name in tqdm(csv_files, bar_format='{l_bar}\033[32m{bar}\033[0m{r_bar}'):
        df_columns = get_columns_from_file(name, COLUMN_SETS_ABP) # check USA format
        if df_columns is None:
            df_columns = get_columns_from_file(name, COLUMN_SETS_ABP, sep=';', decimal=',') # check European format
        if df_columns is None:
            with open(dir_to_write_log, 'w') as file:
                file.write('{}_{}_analysis \n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name.name))
            continue
        else:
            df_abp, df_co2, df_EtCO2 = df_columns
        
        name_without_csv = os.path.splitext(os.path.basename(name))[0]  # getting the name of the file without the .csv extension
        df_abp = df_abp.dropna().reset_index(drop=True) # resetting the index of the dataframe
        processed_signal = nk.ppg_process(df_abp, sampling_rate=sr)  # processing the signal
        
        hrv = nk.hrv(processed_signal[0], sampling_rate=sr)  # performing the HRV analysis
        interesting_content = hrv[CONTENT]        # freq content needs to be calculated separately because of the different sampling rate
        cleaned_signal = nk.ppg_clean(df_abp, sampling_rate=sr)  # cleaning the signal
        r_peaks = nk.ppg_findpeaks(cleaned_signal, sampling_rate=sr)['PPG_Peaks']  # finding the R-peaks
        pyhrv_freq = pd.DataFrame([fd.welch_psd(rpeaks=r_peaks, show=False)['fft_abs']],
                      columns=FREQ_CONTENT) # using pyhrv library to calculate the frequency domain parameters
        
        mean_signals = pd.DataFrame({
            'mean_abp': [df_abp.mean()],
            "mean_co2": [df_co2.mean()], 
            "mean_EtCO2": [df_EtCO2.mean()]
        })
        
        fund_f, fund_amp = calculate_fundamental_component(cleaned_signal, sr)
        final_fund = pd.DataFrame({
            'fund_f': [fund_f],
            'fund_amp': [fund_amp]
            })
        
        merged_df = pd.concat([interesting_content, pyhrv_freq, final_fund, mean_signals], axis=1)  # merging the dataframes
        merged_df['file_name'] = name_without_csv  # adding file_name column
        final_df_abp = pd.concat([final_df_abp, merged_df],
                                     ignore_index=True)  # concatenate final_df with merged_df
        
    return final_df_abp


def ecg_calculation(csv_files, sr, dir_to_write_log):
    """
    This function takes a list of csv files and returns a dataframe with the calculated HRV parameters for the ECG signal.
    
    Parameters
    ----------
    csv_files : list
        A list of all the csv files in the folder.
    
    Returns
    -------
    final_df_ecg : DataFrame
        A dataframe with the calculated HRV parameters for the ECG signal.
    """
    final_df_ecg = pd.DataFrame()
    
    for name in tqdm(csv_files, bar_format='{l_bar}\033[31m{bar}\033[0m{r_bar}'):       
        # here i need to adjust the code to search for the ECG signal in the csv file
        df_columns = get_columns_from_file(name, COLUMN_SETS_ECG)
        if df_columns is None:
            df_columns = get_columns_from_file(name, COLUMN_SETS_ECG, sep=';', decimal=',')

        if df_columns is None:
            with open(dir_to_write_log, 'a') as file:
                file.write('{}_{}_analysis \n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name.name))            
                continue
        else:
            df_ecg, df_co2, df_EtCO2 = df_columns
        df_ecg = df_ecg.dropna().reset_index(drop=True) 
        processed_signal = nk.ecg_process(df_ecg, sampling_rate=sr)
        hrv = nk.hrv(processed_signal[0], sampling_rate=sr)
        name_without_csv = os.path.splitext(name.name)[0]
        interesting_content = hrv[CONTENT]
        cleaned_signal = nk.ecg_clean(df_ecg, sampling_rate=sr)
        r_peaks = nk.ecg_findpeaks(cleaned_signal, sampling_rate=sr)['ECG_R_Peaks']
        pyhrv_freq = pd.DataFrame([fd.welch_psd(rpeaks=r_peaks, show=False)['fft_abs']],
                          columns=FREQ_CONTENT) 

        mean_signals = pd.DataFrame({
            'mean_ecg': [df_ecg.mean()],
        })
        merged_df = pd.concat([interesting_content, pyhrv_freq, mean_signals], axis=1)
        merged_df['file_name'] = name_without_csv

        final_df_ecg = pd.concat(
                [final_df_ecg, merged_df], ignore_index=True)  # concatenate final_df with merged_df            
             
    return final_df_ecg

def final_data(df_abp, df_ecg, name):
    """
    This function takes two dataframes and returns a dataframe with the calculated HRV parameters for the ABP and ECG signals.

    Parameters
    ----------
    df_abp : DataFrame
        A dataframe with the calculated HRV parameters for the ABP signal.
    df_ecg : DataFrame
        A dataframe with the calculated HRV parameters for the ECG signal.

    Returns
    -------
    final_df : DataFrame
        A dataframe with the calculated HRV parameters for the ABP and ECG signals merged.
    """
    dataframes = {'abp': df_abp, 'ecg': df_ecg}

    for key, df in dataframes.items():
        df.to_csv(f'{name}/{key}_output.csv', index=False)
        dataframes[key] = df.drop_duplicates(subset='file_name')
    final_df = pd.merge(df_abp, df_ecg, on='file_name', suffixes=('_abp', '_ecg'))
    final_df = final_df.drop_duplicates(subset='file_name')
    final_df.to_csv('{}/final_output.csv'.format(name), index=False)
    return final_df


def bland_altman_plot(data1, data2, *args, **kwargs):
    """
    Bland-Altman Plot.
    A Bland-Altman plot is a graphical method to analyze the differences between 
    two methods of measurement. The mean of the two measurements is plotted on 
    the x-axis, and the difference between the two measurements is plotted on the y-axis.

    Parameters
    ----------
    data1, data2 : array-like
        Arrays of data.
    args, kwargs   
        Other arguments are passed to the scatter plot function.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    The Bland-Altman plot is a simple and powerful way to visualize the agreement 
    between two measurements. It is created by plotting the mean of two measurements 
    against the difference between the two measurements. The mean is plotted on the x-axis, 
    and the difference is plotted on the y-axis. The plot also includes the mean difference 
    and the limits of agreement (mean difference ± 1.96 * standard deviation of the differences).
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)  # Mean of the data
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)  # Use mean1 directly as x-values
    plt.axhline(md, color='gray', linestyle='--', label=f'{md:.2f}')  # Mean Difference line
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--', label=f'{np.round(md + 1.96 * sd, decimals=3)}')
    plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label=f'{np.round(md - 1.96 * sd, decimals=3)}')

    # Add legend
    plt.legend()
    plt.xlabel('Mean of the two measurements')
    plt.ylabel('Difference between the two measurements')

def calculate_fundamental_component(signal, fs, low_f=0.66, high_f=3):
    """
    Calculate the fundamental component of a signal using the FFT method.
    Parameters
    ----------
    signal : array-like
        The input signal.
    fs : int
        The sampling frequency of the signal.
    
    Returns
    -------
    fund_f : float
        The frequency of the fundamental component.
    fund_amp : float
        The amplitude of the fundamental component.
    """
    n_fft = len(signal)
    win_fft_amp = sp_fft.rfft(sp_sig.detrend(signal) * sp_sig.windows.hann(n_fft), n=n_fft)
    corr_factor = n_fft / np.sum(sp_sig.windows.hann(n_fft))
    win_fft_amp = abs(win_fft_amp) * 2 / n_fft * corr_factor

    win_fft_f = sp_fft.rfftfreq(n_fft, d=1 / fs)
    f_low = int(low_f * n_fft / fs)
    f_upp = int(high_f * n_fft / fs)
    win_fft_amp_range = win_fft_amp[f_low:f_upp]
    fund_idx = np.argmax(win_fft_amp_range) + f_low

    fund_f = win_fft_f[fund_idx]
    fund_amp = win_fft_amp[fund_idx]

    return fund_f, fund_amp


if __name__ == '__main__':
    sampling_rate = int(os.getenv('SAMPLING_RATE'))
    csv_files, name= os.getenv('PATH')
    dir_name = str(name) + "_hrv_analysis_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df_abp  = abp_calculation(csv_files,sampling_rate, f'{dir_name}/abp_log.txt')
    df_ecg  = ecg_calculation(csv_files, sampling_rate, f'{dir_name}/ecg_log.txt')
    final_df = final_data(df_abp, df_ecg, dir_name)

    for i in CONTENT + FREQ_CONTENT:
        plt.figure(figsize=(10, 6))
        bland_altman_plot(final_df[i + '_abp'], final_df[i + '_ecg'])
        plt.title(f'{i}')
        plt.savefig(f'{dir_name}/Bland_Altman_Plot_{i}.png')
