'''
Script to handle all data processes.
'''
import os

import numpy as np
import pandas as pd
import torch
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path


def preprocess_data(data, timestamps, upsample_fac, normalize_data, scale_val, filter_data, startTrialAtNull):
    """
    Filter, normalize, offset and resample the data.
    Uses Multidimensional image processing (scipy.ndimage) for filtering.
    Uses Signal processing (:mod:`scipy.signal`) for resampling.
    """

    data_steps = len(data[0])
    filter_size = [0, int((data_steps * upsample_fac) /
                          int((len(data[0])/4)+0.5)), 0]  # filter over time

    # up, or downsampling
    if upsample_fac < 1.0:
        data = data[::int(1 / upsample_fac)]
    elif upsample_fac > 1.0:
        data = signal.resample(
            data, int(data_steps * upsample_fac), axis=1)  # upsample
        time_interpolate = interp1d(range(data_steps), timestamps)
        timestamps = time_interpolate(
            np.linspace(0, data_steps - 1, int(data_steps * upsample_fac))
        )

    if startTrialAtNull:
        first_val = data[:, 0, :]
        data = data - first_val[:, None, :]

    # TODO inlcude normalization of data
    # TODO per channel or per trial?
    # TODO include tick box to enable/disable normalization
    # normalize data
    if normalize_data:
        data = (data/np.max(data))*scale_val
    else:
        data = data*scale_val

    if filter_data:
        data = ndimage.uniform_filter(
            data, size=filter_size, mode="nearest"
        )

    return timestamps, data


def split_data(data):
    """
    Split each channel in two (positive, abs(negative)) by adding the 
    absolute value of the negative values as a new channel next to the original.
    """
    data = np.array(data)
    data_split = np.zeros((data.shape[0], data.shape[1], data.shape[2] * 2))
    data_split[:, :, ::2] = np.where(data > 0, data, 0)
    data_split[:, :, 1::2] = abs(np.where(data < 0, data, 0))

    return data_split


def load_data(
    file_name="./data/data_braille_letters_all.pkl",
    upsample_fac=1.0,
    normalize_data=False,
    scale_val=1,
    filter_data=False,
    startTrialAtNull=False
):
    """
    Load sample-based data and apply preprocessing.
    """
    data_dict = pd.read_pickle(file_name)
    label_key = "class"
    data_key = "data"

    # Extract data
    # data.shape: (n_trials, n_timesteps, n_channels) with n_trial = n_classes * n_repetition
    data = np.array([x for x in data_dict[data_key].to_numpy()])
    labels = np.array([x for x in data_dict[label_key].to_numpy()])
    # TODO add timestamp as list to allow differnet data len
    timestamps = np.array([x for x in data_dict["timestamp"].to_numpy()])

    if timestamps.shape[-1] != data.shape[1]:
        # create timestamps for each trial
        timestamps = np.arange(
            0, data.shape[1]*1E-3, 1E-3).repeat(data.shape[0]).reshape(data.shape[0], data.shape[1])
    # if labels are strings, convert to integers
    if isinstance(labels[0], str):
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

    # preprocess data
    timestamps_resampled, data_resampled = preprocess_data(
        data=data, timestamps=timestamps, upsample_fac=upsample_fac, normalize_data=normalize_data, scale_val=scale_val, filter_data=filter_data, startTrialAtNull=startTrialAtNull)

    # split data
    if np.min(data_resampled) < 0:
        data_resampled_split = split_data(data_resampled)
        data_split = torch.as_tensor(data_resampled_split, dtype=torch.float)
    else:
        data_split = None

    data = torch.as_tensor(data_resampled, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.long)

    return data_split, labels, timestamps_resampled, le, data
