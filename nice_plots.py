import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from feature_extraction import *
import seaborn as sns
from scipy.signal import cwt, morlet


def signal_plot(sampled_rows):
    fig, ax = plt.subplots(3, 3, figsize=(13,8))
    pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    i=0

    sns.set_style("ticks")
    for label, row in sampled_rows.iterrows():
        if i==6:
            i=7
        acc = np.array(row[[col for col in row.index if col.startswith('acc')]])
        gyr = np.array(row[[col for col in row.index if col.startswith('gyr')]])
        temp = pd.DataFrame({"value":acc,
                            "time":np.arange(len(acc)),
                            "Device": "accelerometer"})
        temp1 = pd.DataFrame({"value":gyr,
                            "time":np.arange(len(acc)),
                            "Device": "gyroscope"})
        temp = pd.concat([temp1,temp])

        sns.lineplot(data=temp, x="time", y="value", hue="Device", ax = ax[pos[i]], legend="brief")
        ax[pos[i]].set_title(label[0])
        # Remove x and y axis ticks since uninformative
        ax[pos[i]].set_xticks([])
        ax[pos[i]].set_yticks([])
        i+=1

    plt.suptitle("Sample signals for each class")
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.delaxes(ax[pos[6]])
    plt.delaxes(ax[pos[8]])
    plt.show()

    return None


def fft_plot(sampled_rows, device: str):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(3, 3, figsize=(13,8))
    pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    i=0
    for label, row in sampled_rows.iterrows():

        if i==6:
            i=7

        if device=="accelerometer":
            vec = np.array(row[[col for col in row.index if col.startswith('acc')]])
            color="purple"
        else:
            vec = np.array(row[[col for col in row.index if col.startswith('gyr')]])
            color="darkblue"

        fourier_data = fft(vec) # take the magnitudes
        magnitudes = np.abs(fourier_data)
        n = len(vec)
        temp = pd.DataFrame({"fft": magnitudes,
                            "freq": fftfreq(n)})
            
        sns.lineplot(data=temp, x="freq", y="fft", ax = ax[pos[i]], color=color)
        ax[pos[i]].set_title(label[0])
        i+=1

    plt.suptitle(f'Abs(Fast Fourier Transform Magnitude) for sample {device} signals')
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.delaxes(ax[pos[6]])
    plt.delaxes(ax[pos[8]])
    plt.show()

    return None


def maxbin_plot(sampled_rows, device: str):
    sns.set_style("ticks")
    fig, ax = plt.subplots(3, 3, figsize=(13,8))
    pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    i=0

    for label, row in sampled_rows.iterrows():
        if i==6:
            i=7
        if device=="accelerometer":
            vec = np.array(row[[col for col in row.index if col.startswith('acc')]])
            color = "purple"
        else:
            vec = np.array(row[[col for col in row.index if col.startswith('gyr')]])
            color = "darkblue"

        max_ps = fourier_magnitudes(vec, plot=True)
        freq = list(fftfreq(int(len(vec)/2)))

        num_bins = 10
        hist_counts, bin_edges = np.histogram(max_ps[0], bins=num_bins)

        ax[pos[i]].plot(max_ps[0][1:], max_ps[1][1:], marker="o", color=color)
        ax[pos[i]].set_xlabel('Frequency')
        ax[pos[i]].set_ylabel('Power spectrum')
        if device=="accelerometer":
            top = max(max_ps[1][1:])+5e3
        else:
            top = max(max_ps[1][1:])+5e5
        ax[pos[i]].set_ylim(top=top)
        ax[pos[i]].set_title(label[0])
        # Add transparent bands using fill_between for each bin
        for k in range(1, num_bins):
            lower_band = bin_edges[k]
            upper_band = bin_edges[k + 1]
            if k%2==0:
                ax[pos[i]].fill_betweenx([0, max(max_ps[1])], lower_band, upper_band, alpha=0.3, color='lightblue')
            else:
                ax[pos[i]].fill_betweenx([0, max(max_ps[1])], lower_band, upper_band, alpha=0.3, color='gray')
        i+=1

    plt.suptitle(f'Power spectrum peaks for sample {device} signals (excluded peak at freq=0)')
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.delaxes(ax[pos[6]])
    plt.delaxes(ax[pos[8]])
    plt.show()

    return None


def cwt_plot(sampled_rows):
    sns.set_style("ticks")
    fig, ax = plt.subplots(3, 3, figsize=(13,8))
    pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    i=0

    for label, row in sampled_rows.iterrows():
        if i==6:
            i=7
        vec = np.array(row[[col for col in row.index if col.startswith('acc')]])
        centered_acc = vec - np.mean(vec)
        wave = cwt(vec, morlet, [20])
        wave = wave.reshape(len(centered_acc),)
        temp = pd.DataFrame({"value": centered_acc,
                            "time": np.arange(len(centered_acc)),
                            "wave": wave})
        sns.lineplot(data=temp, x="time", y="wave", ax = ax[pos[i]], color="plum")
        sns.lineplot(data=temp, x="time", y="value", ax = ax[pos[i]], color="purple")
        ax[pos[i]].set_title(label[0])
        # Remove x and y axis ticks since uninformative
        ax[pos[i]].set_xticks([])
        ax[pos[i]].set_yticks([])
        i+=1

    plt.suptitle("Wavelet Continuous Transform for sample Accelerometer signals")
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.delaxes(ax[pos[6]])
    plt.delaxes(ax[pos[8]])
    plt.show()

    return None