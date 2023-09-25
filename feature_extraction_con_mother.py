
from typing import List, Union
import pandas as pd
import numpy as np
import re
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import statsmodels.api as sm


def flatten_ts(data):
    ''' The function flattens the signals (6 signals) belonging to the same time slot
    to the same row in the new data frame. It returns this new dataframe with a row for
    each flattened vector containing 6 timeseries (3 for the accelerometer and 3 for the
    gyrocscope) of 400 values each. Then each row presents its corresponding signal label
    '''
    # every row is an observation of length 400 x 3 x 2
    # Split the list into chunks of 400 values
    chunk_size = 400

    new_data = pd.DataFrame()
    # Iterate over the original columns
    for col in data.columns:
        if col != "label":
            # Extract values from the column and reshape into chunks of 400 rows
            column_values = data[col].values

            chunks = [column_values[i:i + chunk_size] for i in range(0, len(column_values), chunk_size)]

            # Create a DataFrame
            tmp = pd.DataFrame(chunks, columns=[f'{col}_{i+1}' for i in range(chunk_size)])
            new_data = pd.concat([new_data, tmp], axis=1)

    labels = np.array(data.label.iloc[np.arange(0,data.shape[0], 400)])
    new_data["label"] = labels
    return new_data


######## set of functions useful for applying a transformation to our data frame in an efficient way ############

def chunk_splitting(row, dim=2400):

    # Split the list into chunks of 400 values
    n_chunks = dim/400
    chunks = np.array_split(np.array(row.values[:len(row.values)-1], dtype=float), n_chunks)

    return chunks

def acc_sum(vec):
    acc = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    
    return acc

def gyr_sum(vec):
    gyr = np.sqrt(vec[3]**2 + vec[4]**2 + vec[5]**2)

    return gyr


def vec_sum(new_data):
    df = pd.DataFrame()
    chunk_size = 400
    # horizontally split each row in timeseries of length 400
    chunks = new_data.apply(lambda x: chunk_splitting(x), axis=1)
    # apply the vectorial sum for acc and gyr
    new_data["acc_sum"] = chunks.apply(lambda x: acc_sum(x))
    new_data["gyr_sum"] = chunks.apply(lambda x: gyr_sum(x))
    # rename the columns
    df[[f'acc_{i+1}' for i in range(chunk_size)]] = pd.DataFrame(new_data.acc_sum.to_list(), index = new_data.index)
    df[[f'gyr_{i+1}' for i in range(chunk_size)]] = pd.DataFrame(new_data.gyr_sum.to_list(), index = new_data.index)
    labels = new_data["label"]
    return df, labels


def vec_sum2(new_data):
    df = pd.DataFrame()
    chunk_size = 400
    # horizontally split each row in timeseries of length 400
    chunks = new_data.apply(lambda x: chunk_splitting2(x), axis=1)
    # apply the vectorial sum for acc and gyr
    new_data["acc_sum"] = chunks.apply(lambda x: acc_sum(x))
    
    # rename the columns
    df[[f'acc_{i+1}' for i in range(chunk_size)]] = pd.DataFrame(new_data.acc_sum.to_list(), index = new_data.index)
    
    labels = new_data["label"]
    return df, labels


###############################################################################################################


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


# def psd_stats(signal: np.ndarray) -> np.ndarray:
#     """
#     This function takes a matrix (signal) which contains a time series for each row.
#     It calculates the Power Spectral Density (PSD) from the signal and returns three different summary statistics
#     from the distribution of PSD: the median, the mean absolute deviation, and the skewness (third moment).

#     Args:
#         signal (np.ndarray): The input matrix, where each row represents a time series.

#     Returns:
#         np.ndarray: An array containing the median, mean absolute deviation, and skewness of the Power Spectral Density.
#     """
#     signal= np.array(signal)
#     # let's calculate the Power Spectral Density
#     psd = np.array(np.abs(np.apply_along_axis(periodogram, 0, signal)[1]), dtype=np.float64)
    
#     # extract our main summaries from the distribution
#     median = np.median(psd, axis=0)
#     # calculate the third moment (skewness)
#     third_moment = skew(psd, axis=0)
#     # calculate the mean absolute deviation
#     mad = np.mean(np.abs(psd - np.mean(psd, axis=0)), axis = 0)
    
#     return np.array([median, mad, third_moment])


# def acf(signal: np.ndarray, n_lags: int = 10) -> np.ndarray:
#     """
#     This function takes a matrix (signal) which contains a time series for each row.
#     It calculates the auto-correlation for different lags.

#     Args:
#         signal (np.ndarray): The input matrix, where each row represents a time series.
#         num_lags (int, optional): The number of lags to calculate the auto-correlation for. Defaults to 10.

#     Returns:
#         np.ndarray: The auto-correlation for each time series in the input matrix.
#     """
#     signal= np.array(signal)
#     autocorrelations = np.apply_along_axis(lambda x: sm.tsa.acf(x, nlags=n_lags), 0, signal)


#     return autocorrelations

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



def preproc(df, labels, n_bins=10, n_lags=10):

    # group timeseries by device (accelerometer and gyroscope)
    group_dict = {}
    for col in df.columns:
        # match "acc" or "gyr"
        match = re.match("^([a-z]{3})", col)
        group_dict[col] = match.group() if match else None

    new_df = df.T.groupby(group_dict, axis=0)

    # evaluate max magnitudes (intesities of the frequencies from Fourier Fast transform) for each observation for each accelerometer
    magns = new_df.apply(lambda x: fourier_magnitudes(x, n_bins))

    # # evaluate autocorrelations
    # # autocorrs = new_df.apply(lambda x: acf(x, n_lags))
    # # evaluate psd stats
    # # psds = new_df.apply(psd_stats)

    # adjust in DataFrame format
    magns = adjust_df(magns)
    # autocorrs = adjust_df(autocorrs)
    # psds = adjust_df(psds)

    # change col names of our dataframe of magnitudes
    existing_columns = magns.columns
    # generate a list of new column names
    new_columns = []
    i=1
    for j,col_name in enumerate(existing_columns):
        if ((j+1)%(n_bins+1)) == 0:
            i=1
        new_columns.append(f"{col_name}_max_mag_{i}")
        i+=1
    magns.columns = new_columns
    new_df = magns
    # # change col names of our dataframe of autocorrelations
    # existing_columns = autocorrs.columns
    # # generate a list of new column names
    # new_columns = []
    # i=0
    # for j,col_name in enumerate(existing_columns):
    #     if ((j+1)%(n_lags+2)) == 0:
    #         i=0
    #     new_columns.append(f"{col_name}_acf_l{i}")
    #     i+=1
    # autocorrs.columns = new_columns

    # # remove all the column which correspond to autocorrelation with lag=0
    # autocorrs = autocorrs.loc[:, (autocorrs != 1).any(axis=0)]

    # # change col names of our dataframe of autocorrelations
    # existing_columns = psds.columns
    # # generate a list of new column names
    # new_columns = []
    # i=1
    # n_stats = 3
    # for j,col_name in enumerate(existing_columns):
    #     new_columns.append(f"{col_name}_psd{i}")
    #     if ((j+1)%(n_stats)) == 0:
    #         i=0
    #     i+=1
    # psds.columns = new_columns
    
    # new_df = pd.concat([magns, autocorrs, psds], axis=1)

    new_df["label"] = labels

    return new_df