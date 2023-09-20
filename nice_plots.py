import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from feature_extraction import *

def plot_magnitudes(sample_ts):
    fourier_data = fft(sample_ts) # take the magnitudes
    magnitudes = np.abs(fourier_data)**2
    n = len(sample_ts)
    freq = fftfreq(n)
    plt.plot(freq, magnitudes)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title("Example of Magnitudes for each frequency of a timeseries")
    # Add the symmetry axis.
    plt.axvline(x=0, color='purple', linestyle='--', label='Symmetry Axis')
    plt.text(0, 15e4, 'Symmetry Axis', rotation=90, va='center', ha='right', color='purple')
    plt.grid()
    plt.show()


def plot_peaks(sample_ts):
    max_ps = fourier_magnitudes(sample_ts, plot=True)
    freq = list(fftfreq(int(len(sample_ts)/2)))

    num_bins = 10
    hist_counts, bin_edges = np.histogram(max_ps[0], bins=num_bins)

    plt.plot(max_ps[0], max_ps[1], marker="o")
    plt.xlabel('Frequency')
    plt.ylabel('Power spectrum')
    plt.title("Power spectrum peaks for several bins")

    # Add transparent bands using fill_between for each bin
    for i in range(num_bins):
        lower_band = bin_edges[i]
        upper_band = bin_edges[i + 1]
        if i%2==0:
            plt.fill_betweenx([0, max(max_ps[1])], lower_band, upper_band, alpha=0.3, color='lightblue')
        else:
            plt.fill_betweenx([0, max(max_ps[1])], lower_band, upper_band, alpha=0.3, color='gray')

    plt.show()