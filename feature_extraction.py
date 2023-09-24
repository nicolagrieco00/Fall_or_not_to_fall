
from typing import List, Union
import pandas as pd
import numpy as np
import re
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet


##### FLATTEN THE DATA SET IN AN AD HOC DATA FRAME ##########################################################################################

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


######## FUNCTIONS FOR VECTORIAL SUM (INTENSITY OF VECTORS IN 3-D SPACE) ####################################################################

def chunk_splitting(row):

    # Split the list into chunks of 400 values
    n_chunks = 2400/400
    chunks = np.array_split(np.array(row.values[:len(row.values)-1], dtype=float), n_chunks)

    return chunks

def chunk_splitting2(row):
    # Split the list into chunks of 400 values
    n_chunks = 1200/400
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

<<<<<<< Updated upstream
=======

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
>>>>>>> Stashed changes

##### FAST FOURIER TRANSFORM STATISTICS FEATURE EXTRACTION #############################################################################

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

##### FUNCTIONS FOR PEAKS WAVELETS MAX COEFFICIENTS #####################################################################################

def cwt_coeff(peak_wave, mother_wave, scale, translation):
    std_mother_wave = (mother_wave - translation)/scale
    coeff = sum(peak_wave * std_mother_wave)/np.sqrt(scale)
    return coeff

def max_cwt_coeff(peak_wave, mother_wave):
    max_coeff = np.NINF
    translation_vec = np.arange(10, 30, 10)
    for b in translation_vec:
        new_coeff = cwt_coeff(peak_wave, mother_wave, 2, b)
        if new_coeff > max_coeff:
            max_coeff = new_coeff
    return max_coeff

def peakes_wavelet_approx(signal, mother_wave, plot=False, label=None):
    half_window = 20 # number of sampled values in our data collection in 2 seconds of time
    threshold = 1.9 * 9.81 
    peaks = signal[signal >= threshold]

    if len(peaks) == 0:
        if plot:
            print("There are no peaks!")
        return 0
    indexes = np.where(signal >= threshold)[0]
    peaks_df = pd.DataFrame({"peak": peaks, "idx": np.array(indexes, dtype=int)})
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
    for row in peaks_df.iterrows():
        idx = int(row[1].idx)
        # take the values inside the window centered around the actual peak
        # but first check if the peak is in extreme positions of the signal
        if (idx - half_window) < 0:
            temp = signal[: (idx + half_window)]
        elif (idx + half_window) > len(signal):
            temp = signal[(idx - half_window) :]
        else:
            temp = signal[(idx - half_window): (idx + half_window)]

        # wavelet cont transform of the peak pattern
        peak_wave = cwt(temp, morlet, [18])
        peak_wave = peak_wave.reshape(peak_wave.shape[1],)
        max_cwt = max_cwt_coeff(peak_wave, mother_wave[:len(peak_wave)])
        x = np.arange(0,len(temp))
        # temp = pd.DataFrame({"arr": arr, "wave":wave})
        if plot:
            ax[1].plot(x,peak_wave)
            ax[0].set_title("Peaks signal pattern")
            ax[0].plot(x,temp)
            ax[1].set_title("Peaks Wavelet Transform")
    if plot:
        ax[1].plot(x,mother_wave[:len(x)], color="darkblue", linewidth=4, alpha=0.7)
        plt.suptitle(f'{label.upper()} signal   -   CWD coeff.: {round(abs(max_cwt),3)}')
        plt.show()
    return max_cwt

def acc_max_cwt(row, mother_wave):
    row = np.array(row[:400])
    max_coeff = peakes_wavelet_approx(row, mother_wave)
    return max_coeff