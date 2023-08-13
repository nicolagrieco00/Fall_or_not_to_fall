
from typing import List, Union
import pandas as pd
import numpy as np
import re
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import statsmodels.api as sm
from scipy.stats import skew
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression


def maxbin(row: Union[List[float], np.ndarray], plot: bool, n_bins: int = 10) -> List[float]:
    """
    Divide the data into a specified number of bins and calculate the maximum value for each bin.

    Args:
        row (list or array): The data to be divided into bins.
        num_bins (int, optional): The number of bins to divide the data into. Defaults to 10.

    Returns:
        list: A list containing the maximum value for each bin.
    """
    n = len(row)
    freq = list(fftfreq(n*2))
    step = n//n_bins
    max_values = []
    max_freqs = []
    for i in range(0, n+1, step):
        if i==0:
            prec = i
            continue
        mm = max(row[prec: i])
        max_values.append(mm)
        max_freqs.append(freq[list(row).index(mm)])
        prec = i
    if plot:
        return max_freqs, max_values
    else:
        return max_values


def fourier_magnitudes(signal: np.ndarray, n_bins: int = 10, plot: bool = False) -> np.ndarray:
    """
    This function takes a matrix (signal) which contains a time series for each row.
    It applies the Fast Fourier Transform and uses the get_max_per_bin function to find the maximum peaks for each bin of the arrays of the power spectrum for the FFT.

    Args:
        signal (np.ndarray): The input matrix, where each row represents a time series.
        num_bins (int, optional): The number of bins to divide the data into. Defaults to 10.

    Returns:
        np.ndarray: The maximum peaks for each bin of the arrays of the power spectrum for the FFT.
    """
    signal= np.array(signal)
    # let's calculate the FFT
    magnitudes = np.apply_along_axis(fft, 0, signal) 
    # extract just the first half of power spectrum and discover the peaks
    if len(magnitudes.shape)==1:
        magnitudes=magnitudes[:((magnitudes.shape[0])//2)]
    else:
        magnitudes=magnitudes[:((magnitudes.shape[0])//2),:]
    # apply power spectrum formula
    magnitudes = np.abs(magnitudes)**2 
    # take the peaks for each bin
    peaks = np.apply_along_axis(lambda x: maxbin(x, plot, n_bins), 0, magnitudes)
    
    return peaks


def psd_stats(signal: np.ndarray) -> np.ndarray:
    """
    This function takes a matrix (signal) which contains a time series for each row.
    It calculates the Power Spectral Density (PSD) from the signal and returns three different summary statistics
    from the distribution of PSD: the median, the mean absolute deviation, and the skewness (third moment).

    Args:
        signal (np.ndarray): The input matrix, where each row represents a time series.

    Returns:
        np.ndarray: An array containing the median, mean absolute deviation, and skewness of the Power Spectral Density.
    """
    signal= np.array(signal)
    # let's calculate the Power Spectral Density
    psd = np.array(np.abs(np.apply_along_axis(periodogram, 0, signal)[1]), dtype=np.float64)
    
    # extract our main summaries from the distribution
    median = np.median(psd, axis=0)
    # calculate the third moment (skewness)
    third_moment = skew(psd, axis=0)
    # calculate the mean absolute deviation
    mad = np.mean(np.abs(psd - np.mean(psd, axis=0)), axis = 0)
    
    return np.array([median, mad, third_moment])


def acf(signal: np.ndarray, n_lags: int = 10) -> np.ndarray:
    """
    This function takes a matrix (signal) which contains a time series for each row.
    It calculates the auto-correlation for different lags.

    Args:
        signal (np.ndarray): The input matrix, where each row represents a time series.
        num_lags (int, optional): The number of lags to calculate the auto-correlation for. Defaults to 10.

    Returns:
        np.ndarray: The auto-correlation for each time series in the input matrix.
    """
    signal= np.array(signal)
    autocorrelations = np.apply_along_axis(lambda x: sm.tsa.acf(x, nlags=n_lags), 0, signal)

    return autocorrelations


def adjust_df(s: pd.Series) -> pd.DataFrame:
    """
    This function takes a pandas Series, explodes and transposes it, and returns it as a DataFrame.

    Args:
        series (pd.Series): The input pandas Series.

    Returns:
        pd.DataFrame: The adjusted DataFrame.
    """
    df_list = [pd.DataFrame(matrix, index=[index]*len(matrix)) for index, matrix in s.items()]
    new_df = pd.concat(df_list, axis=0)
    new_df.columns = range(new_df.shape[1])
    new_df = new_df.T
    return new_df


def get_season(date):
    if date.month in [3, 4, 5]:
        return 1 # spring
    elif date.month in [6, 7, 8]:
        return 2 # summer
    elif date.month in [9, 10, 11]:
        return 3 # autumn
    else:
        return 4 # winter
    

def preproc(df, n_bins=10, n_lags=10):
    
    # group timeseries by accelerometer (device)
    group_dict = {}
    for col in df.columns:
        match = re.match("^([A-Z1-9]{3})", col)
        group_dict[col] = match.group() if match else None

    new_df = df.T.groupby(group_dict, axis=0)

    # evaluate max magnitudes (intesities of the frequencies from Fourier Fast transform) for each observation for each accelerometer
    magns = new_df.apply(lambda x: fourier_magnitudes(x, n_bins))
    # evaluate autocorrelations
    autocorrs = new_df.apply(lambda x: acf(x, n_lags))
    # evaluate psd stats
    psds = new_df.apply(psd_stats)

    # adjust in DataFrame format
    magns = adjust_df(magns)
    autocorrs = adjust_df(autocorrs)
    psds = adjust_df(psds)

    # change col names of our dataframe of magnitudes
    existing_columns = magns.columns
    # generate a list of new column names
    new_columns = []
    i=1
    for j,col_name in enumerate(existing_columns):
        if ((j+1)%(n_bins+1)) == 0:
            i=1
        new_columns.append(f"{col_name}_mag_max_{i}")
        i+=1
    magns.columns = new_columns

    # change col names of our dataframe of autocorrelations
    existing_columns = autocorrs.columns
    # generate a list of new column names
    new_columns = []
    i=0
    for j,col_name in enumerate(existing_columns):
        if ((j+1)%(n_lags+2)) == 0:
            i=0
        new_columns.append(f"{col_name}_acf_l{i}")
        i+=1
    autocorrs.columns = new_columns

    # remove all the column which correspond to autocorrelation with lag=0
    autocorrs = autocorrs.loc[:, (autocorrs != 1).any(axis=0)]

    # change col names of our dataframe of autocorrelations
    existing_columns = psds.columns
    # generate a list of new column names
    new_columns = []
    i=1
    n_stats = 3
    for j,col_name in enumerate(existing_columns):
        new_columns.append(f"{col_name}_psd{i}")
        if ((j+1)%(n_stats)) == 0:
            i=0
        i+=1
    psds.columns = new_columns

    # create new column 'Season' based on 'when'
    df['when'] = pd.to_datetime(df['when'], format='%Y-%m')
    df["Season"] = df['when'].apply(get_season)

    # take the number of moths passed from the 'zero' time
    min_date =  df['when'].min()
    df['elapsed_time'] = (df['when'].dt.year - min_date.year) * 12 + (df['when'].dt.month - min_date.month)
    
    new_df = pd.concat([df[["Season", "elapsed_time"]], magns, autocorrs, psds], axis=1)

    return new_df