
from typing import List, Union
import pandas as pd
import os
import numpy as np
import re
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet
from scipy.stats import skew
import warnings


##### FLATTEN THE DATA SET IN AN AD HOC DATA FRAME ##########################################################################################

def flatten_ts(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens time series data into a single row per time slot.

    Args:
        data (pd.DataFrame): Dataframe containing time series data with multiple signals.

    Returns:
        pd.DataFrame: A new dataframe where each row represents a flattened vector containing 
                      time series data for accelerometer and gyroscope.
    """
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

def chunk_splitting(row: pd.Series, dim: int = 2400) -> List[np.ndarray]:
    """
    Splits a row into chunks of a specified size.

    Args:
        row (pd.Series): Row from a DataFrame.
        dim (int): Total dimension to be split into chunks. Default is 2400.

    Returns:
        List[np.ndarray]: List of np.ndarray chunks.
    """

    n_chunks = dim/400
    chunks = np.array_split(np.array(row.values[:len(row.values)-1], dtype=float), n_chunks)
    return chunks


def acc_sum(vec: List[float]) -> float:
    """
    Calculates the vectorial sum of accelerometer readings.

    Args:
        vec (List[float]): List of accelerometer readings.

    Returns:
        float: Vectorial sum of the given readings.
    """

    acc = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return acc


def gyr_sum(vec: List[float]) -> float:
    """
    Calculates the vectorial sum of gyroscope readings.

    Args:
        vec (List[float]): List of gyroscope readings.

    Returns:
        float: Vectorial sum of the given readings.
    """

    gyr = np.sqrt(vec[3]**2 + vec[4]**2 + vec[5]**2)
    return gyr


def vec_sum(new_data: pd.DataFrame, both: bool = True, dim: int = 2400) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Applies vectorial sum to accelerometer and gyroscope data.

    Args:
        new_data (pd.DataFrame): Dataframe containing time series data.
        both (bool): Flag to indicate if both accelerometer and gyroscope sums are to be calculated. Default is True.
        dim (int): Dimension for chunk splitting. Default is 2400.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the processed DataFrame and labels.
    """

    df = pd.DataFrame()
    chunk_size = 400
    # horizontally split each row in timeseries of length 400
    chunks = new_data.apply(lambda x: chunk_splitting(x, dim), axis=1)

    # apply the vectorial sum for acc and gyr
    new_data["acc_sum"] = chunks.apply(lambda x: acc_sum(x))
    # rename the columns
    df[[f'acc_{i+1}' for i in range(chunk_size)]] = pd.DataFrame(new_data.acc_sum.to_list(), index = new_data.index)

    if both:
        new_data["gyr_sum"] = chunks.apply(lambda x: gyr_sum(x))
        df[[f'gyr_{i+1}' for i in range(chunk_size)]] = pd.DataFrame(new_data.gyr_sum.to_list(), index = new_data.index)
    labels = new_data["label"]
    return df, labels

##### MOTHER WAVELET BUILDING ##########################################################################################################


def build_auxiliary_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds an auxiliary dataframe for further processing.

    Args:
        df (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: Processed dataframe.
    """

    const = 9.81
    ## Adding the genric label fall
    df["label"] = "fall"
    ## Drop the column Time(s)
    df_new = df.drop(["Time(s)"], axis = 1)
    ## Flatting the data to use the previous functions
    df_new_flatted = flatten_ts(df_new)

    ## Calculating the magnitude of the acceleration to obtain the final time series
    warnings.filterwarnings('ignore')
    df_new, labels_FF = vec_sum(df_new_flatted, both=False, dim=1200)
    df_new["label"] = labels_FF
    ## moltiply per 9.81
    df_new.iloc[:, :400] = df_new.iloc[:, :400] * const

    return df_new


def find_max_peak(signal: np.ndarray) -> pd.DataFrame:
    """
    Finds the maximum peak in a given signal.

    Args:
        signal (np.ndarray): The input signal array.

    Returns:
        pd.DataFrame: DataFrame containing the maximum peak value and its index.
    """

    max_value = np.max(signal)
    max_index = np.argmax(signal)
    peak_df = pd.DataFrame({"peak": [max_value], "idx": [max_index]})
    return peak_df


def take_peak(signal: np.ndarray) -> np.ndarray:
    """
    Extracts a segment of the signal around its maximum peak.

    Args:
        signal (np.ndarray): The input signal array.

    Returns:
        np.ndarray: A segment of the signal centered around its maximum peak.
    """

     half_window = 20
     peak_df = find_max_peak(signal)
     idx = peak_df["idx"].iloc[0]
     # take the values inside the window centered around the actual peak
     # but first check if the peak is in extreme positions of the signal
     if (idx - half_window) < 0:
        temp = signal[: (idx + half_window)]
     elif (idx + half_window) > len(signal):
        temp = signal[(idx - half_window) :]
     else:
         temp = signal[(idx - half_window):(idx + half_window)]
       
     return temp       


def mean_peak_pattern(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates the mean peak pattern of time series data.

    Args:
        df (pd.DataFrame): DataFrame containing time series data.

    Returns:
        np.ndarray: An array representing the mean peak pattern.
    """

    result_dataset = pd.DataFrame()

    # Iterate through each row of df_FB_new and apply build_mother
    for row in range(len(df.iloc[:, :400])):  # Iterate only over the first 400 columns
        result_vector = pd.DataFrame([take_peak(df.iloc[row,0:400])])  # Apply build_mother to the row
        result_vector.columns = [f'Col_{i+1}' for i in range(40)]
        result_dataset = pd.concat([result_dataset,result_vector], ignore_index= True, axis=0)

    return result_dataset.mean(axis=0).to_numpy()


def mother_wavelet(res: List[pd.DataFrame]) -> np.ndarray:
    """
    Computes the mean of the mother wavelet transformations of the provided data.

    Args:
        res (List[pd.DataFrame]): List of DataFrames representing different categories of data.

    Returns:
        np.ndarray: The mean mother wavelet transformation.
    """

    FF_mean = mean_peak_pattern(res[0])
    FS_mean = mean_peak_pattern(res[1])
    FB_mean = mean_peak_pattern(res[2])
    FF_mean_wave = cwt(FF_mean, morlet, [18])[0]
    FS_mean_wave = cwt(FS_mean, morlet, [18])[0]
    FB_mean_wave = cwt(FB_mean, morlet, [18])[0]

    return np.mean([FF_mean_wave, FS_mean_wave, FB_mean_wave], axis=0)


def extract_wave_df(dir_name: str) -> np.ndarray:
    """
    Extracts wavelet data from files in a specified directory.

    Args:
        dir_name (str): Name of the directory containing the files.

    Returns:
        np.ndarray: The mean wavelet data extracted from the files.
    """

    # Define the directory where your files are located
    directory = os.getcwd() + f'\\{dir_name}\\'

    # Create empty DataFrames for each category (FB, FF, FS)
    df_FB = pd.DataFrame()
    df_FF = pd.DataFrame()
    df_FS = pd.DataFrame()

    # Define the common code to extract 400 rows from a DataFrame
    def extract_400_rows(df):
        desired_row_count = 400
        current_row_count = len(df)
        step_size = max(1, current_row_count // desired_row_count)
        selected_indices = range(0, current_row_count, step_size)
        selected_rows = df.iloc[selected_indices]
        return selected_rows.head(desired_row_count)

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        
        if filename.endswith(".csv"):
            # Determine the category based on the file name
            category = filename.split("_")[-1].split(".")[0]
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(directory, filename), header=None)
            
            # Extract 400 rows using the common code
            selected_rows = extract_400_rows(df)
        
            # Append the selected rows to the appropriate DataFrame based on the category
            if category == 'FB':
                df_FB = pd.concat([df_FB, selected_rows], ignore_index=True)

            elif category == 'FF':
                df_FF = pd.concat([df_FF, selected_rows], ignore_index=True)
                
            elif category == 'FS':
                df_FS = pd.concat([df_FS, selected_rows], ignore_index=True)
            

    df_FF.columns = ["Acc_x", "Acc_y", "Acc_z", "Time(s)"]
    df_FS.columns = ["Acc_x", "Acc_y", "Acc_z", "Time(s)"]
    df_FB.columns = ["Acc_x", "Acc_y", "Acc_z", "Time(s)"]

    # build the auxiliary data frames to build the mother wavelets
    df_FF_new = build_auxiliary_df(df_FF)
    df_FS_new = build_auxiliary_df(df_FS)
    df_FB_new = build_auxiliary_df(df_FB)


    FF_mean = mean_peak_pattern(df_FF_new)
    FS_mean = mean_peak_pattern(df_FS_new)
    FB_mean = mean_peak_pattern(df_FB_new)
    FF_mean_wave = cwt(FF_mean, morlet, [18])[0]
    FS_mean_wave = cwt(FS_mean, morlet, [18])[0]
    FB_mean_wave = cwt(FB_mean, morlet, [18])[0]

    return np.mean([FF_mean_wave, FS_mean_wave, FB_mean_wave], axis=0)


##### FAST FOURIER TRANSFORM STATISTICS FEATURE EXTRACTION #############################################################################

def maxbin(row: Union[List[float], np.ndarray], plot: bool, precision: int = -1, n_bins: int = 10) -> List[float]:
    """
    Calculates the maximum value in each bin of the data.

    Args:
        row (Union[List[float], np.ndarray]): Data to be binned.
        plot (bool): Whether to plot the results.
        precision (int): Number of peaks to consider around the origin. Default is -1 (all peaks).
        n_bins (int): Number of bins. Default is 10.

    Returns:
        List[float]: Maximum value in each bin.
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
        return max_values[:(precision + 1)]


def fourier_magnitudes(signal: np.ndarray, n_bins: int = 10, precision: int = -1, plot: bool = False) -> np.ndarray:
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
    peaks = np.apply_along_axis(lambda x: maxbin(x, plot, precision, n_bins), 0, magnitudes)
    
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


##### FUNCTIONS FOR PEAKS WAVELETS MAX COEFFICIENTS #####################################################################################

def cwt_coeff(peak_wave: np.ndarray, mother_wave: np.ndarray, scale: float, translation: float) -> float:
    """
    Calculates the continuous wavelet transform coefficient for a peak wave.

    Args:
        peak_wave (np.ndarray): The peak wave array.
        mother_wave (np.ndarray): The mother wave array.
        scale (float): The scale factor for the wavelet transform.
        translation (float): The translation factor for the wavelet transform.

    Returns:
        float: The continuous wavelet transform coefficient.
    """

    std_mother_wave = (mother_wave - translation)/scale
    coeff = sum(peak_wave * std_mother_wave)/np.sqrt(scale)
    return coeff


def max_cwt_coeff(peak_wave: np.ndarray, mother_wave: np.ndarray) -> float:
    """
    Finds the maximum continuous wavelet transform coefficient for a given peak wave.

    Args:
        peak_wave (np.ndarray): The peak wave array.
        mother_wave (np.ndarray): The mother wave array.

    Returns:
        float: The maximum continuous wavelet transform coefficient.
    """

    max_coeff = np.NINF
    translation_vec = np.arange(10, 30, 10)
    for b in translation_vec:
        new_coeff = cwt_coeff(peak_wave, mother_wave, 2, b)
        if new_coeff > max_coeff:
            max_coeff = new_coeff
    return max_coeff

def peakes_wavelet_approx(signal: np.ndarray, mother_wave: np.ndarray, plot: bool = False, label: Optional[str] = None) -> float:
    """
    Approximates the wavelet of the peaks in a signal.

    Args:
        signal (np.ndarray): The signal array.
        mother_wave (np.ndarray): The mother wave array.
        plot (bool): Flag to plot the results. Default is False.
        label (Optional[str]): Optional label for plotting. Default is None.

    Returns:
        float: The wavelet approximation of the peaks.
    """

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

def acc_max_cwt(row: Union[List[float], np.ndarray], mother_wave: np.ndarray) -> float:
    """
    Calculates the maximum continuous wavelet transform coefficient for accelerometer data.

    Args:
        row (Union[List[float], np.ndarray]): The accelerometer data.
        mother_wave (np.ndarray): The mother wave array.

    Returns:
        float: The maximum continuous wavelet transform coefficient for the given data.
    """

    row = np.array(row[:400])
    max_coeff = abs(peakes_wavelet_approx(row, mother_wave))
    return max_coeff


##### FINAL FUNCTION FOR PREPROCESSING ################################################################################################

def preproc(data: pd.DataFrame, n_bins: int = 10, precision: int = 4) -> pd.DataFrame:
    """
    Performs preprocessing on the given data including flattening, vector sums, Fourier transforms, and more.

    Args:
        data (pd.DataFrame): The input dataframe.
        n_bins (int): Number of bins for the Fourier transforms. Default is 10.
        precision (int): Precision for the Fourier transforms. Default is 4.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """


    flat_data = flatten_ts(data)
    df, labels = vec_sum(flat_data)

    mother_wave = extract_wave_df("file_csv")
    cwt_coeff = df.apply(lambda x: acc_max_cwt(x, mother_wave), axis=1)

    # group timeseries by device (accelerometer and gyroscope)
    group_dict = {}
    for col in df.columns:
        # match "acc" or "gyr"
        match = re.match("^([a-z]{3})", col)
        group_dict[col] = match.group() if match else None

    new_df = df.T.groupby(group_dict, axis=0)

    # evaluate max magnitudes (intesities of the frequencies from Fourier Fast transform) for each observation for each accelerometer
    magns = new_df.apply(lambda x: fourier_magnitudes(x, n_bins, precision))

    # evaluate psd stats
    psds = new_df.apply(psd_stats)

    # adjust in DataFrame format
    magns = adjust_df(magns)
    psds = adjust_df(psds)

    # change col names of our dataframe of magnitudes
    existing_columns = magns.columns
    numb = len(existing_columns)/2
    # generate a list of new column names
    new_columns = []
    i=1
    for j,col_name in enumerate(existing_columns):
        if ((j+1)%(numb +1)) == 0:
            i=1
        new_columns.append(f"{col_name}_max_mag_{i}")
        i+=1
    magns.columns = new_columns

    # change col names of our dataframe of psd
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
    
    new_df = pd.concat([magns, psds], axis=1)
    new_df["cwt_coeff"] = cwt_coeff
    new_df["label"] = labels

    return new_df