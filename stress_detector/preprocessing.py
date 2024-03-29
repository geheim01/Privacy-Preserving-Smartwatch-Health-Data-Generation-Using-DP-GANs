import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy

from stress_detector.data.datatype import DataType
from synthesizers.preprocessing.wesad import WESADDataset
from synthesizers.utils.preprocessing import get_max_value_from_list, most_common

from . import constants

# 1. Creating the windows
# 2. Create subwindows from the windows
# 3. Calculate the fft of the subwindows
# 4. Average the subwindows


# most frequent element in list
def most_common(lst):
    return max(set(lst), key=lst.count)


# if stress occurress in time interval return 1
def check_for_stress(lst):
    return max(set(lst))


def sliding_windows(data: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Create a sliding window from physiological measurement data.

    Args:
        data (pd.DataFrame): Pandas DataFrame object containing physiological measurements.
        seq_length (int): The size of the window.

    Returns:
        tuple[np.array, np.array]: Windows with physiological measurements and corresponding labels.
    """
    # Exclude 'time_iso' column
    container = data.loc[:, data.columns != "time_iso"]
    containerArray = np.array(container)

    window_size = 60  # Window size in seconds
    overlap_size = 30  # Overlap size in seconds

    # Calculate the number of data points per window and overlap
    window_points = window_size * 1  # Since the capturing frequency is 1 Hz
    overlap_points = overlap_size * 1

    windows = []
    labels = []

    for i in range(0, containerArray.shape[0] - window_points + 1, overlap_points):
        window = containerArray[i : i + window_points]
        windows.append(window)

        label = int(most_common(data["label"][i : i + window_points].to_list()))
        labels.append(label)

    return np.array(windows), np.array(labels)


def create_windows(df: pd.DataFrame, fs: int) -> Tuple[np.ndarray, list]:
    """Creates windows from the dataframe and returns the windows and the labels.
    If the window is assigned to multiple labels, the most common label is chosen for that period.

    Args:
        df (pd.DataFrame): Subject DataFrame
        fs (int): Samples per second

    Returns:
        tuple[np.ndarray,list]: Windows representing the activity of the subject in one minute and the corresponding labels.
    """
    # Create an empty list for the windows and labels
    windows = []
    labels = []

    # Calculate the window length in samples
    window_len = fs * 60

    # Loop over the rows in the DataFrame to create the windows
    for i in range(0, df.shape[0] - window_len, window_len):
        # Get the window data and label
        window = df[i : i + window_len]
        label = int(most_common(df["label"][i : i + window_len].to_list()))

        # Convert the window data to a numpy array
        window = window.to_numpy()

        # Add the window and label to the list
        windows.append(window)
        labels.append(label)

    # Convert the windows and labels to numpy arrays
    windows = np.array(windows)
    labels = np.array(labels)

    # Return the windows and labels as a tuple
    return windows, labels


def create_subwindows(
    window: np.array, signal_subwindow_len: int, signal_name: str, fs: int
) -> np.array:
    """The function creates subwindows from the windows.

    Args:
        df (pd.DataFrame): Windows representing the activity of the subject in one minute.
        signal_subwindow_len (int): Length of the subwindows.
        signal_name (str): Name of the signal.
        fs (int): Samples per second

    Returns:
        list: Subwindows of the signal in the window.
    """

    subwindow_len = (
        fs * signal_subwindow_len
    )  # fs = 64 and sub-window length in seconds = 30
    window_len = fs * 60  # fs = 64 and window length in seconds = 60
    window_shift = (
        1 if fs < 4 else int(fs * 0.25)
    )  # fs = 64 and window shift in seconds = 0.25

    subwindows = np.asarray(
        [
            window[i : i + subwindow_len]
            for i in range(0, window_len - subwindow_len + 1, window_shift)
        ]
    )

    return subwindows


def fft_subwindows(subwindows: list, duration: int, fs: int) -> list:
    """Calculates the fft of the subwindows.

    Args:
        subwindows (list): C
        duration (int): _description_
        f_s (int): _description_

    Returns:
        list: Fft coefficients of the subwindows.
    """
    freqs = []
    yfs = []
    for subwindow in subwindows:
        y = np.array(subwindow)
        yf = scipy.fft.fft(y)
        l = len(yf)
        N = fs * duration
        freq = scipy.fft.fftfreq(N, 1 / fs)

        l //= 2
        amps = np.abs(yf[0:l])
        freq = np.abs(freq[0:l])

        # Sort descending amp
        p = amps.argsort()[::-1]
        freq = freq[p]
        amps = amps[p]

        freqs.append(freq)
        yfs.append(amps)
    return np.asarray(freqs), np.asarray(yfs)


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


def average_window(subwindows_fft: list) -> list:
    """Calculates the average of the fft coefficients of the subwindows.

    Args:
        subwindows_fft (list): List of fft coefficients of the subwindows.

    Returns:
        list: Average of the fft coefficients of the subwindow for signals.
    """
    len_yfs = len(subwindows_fft[0])
    avg_yfs = []
    for i in range(len_yfs):
        i_yfs = []
        for yf in subwindows_fft:
            try:
                i_yfs.append(yf[i])
            except IndexError:
                pass
        avg_yfs.append(sum(i_yfs) / len(i_yfs))
    return avg_yfs


def create_training_data_per_subject(fs, windows):
    X = []
    for i in range(0, len(windows) - 1):
        yfs_averages = []
        for j, signal in enumerate(constants.SIGNAL_SUBWINDOW_DICT.keys()):
            duration_in_sec = constants.SIGNAL_SUBWINDOW_DICT[signal]
            subwindows = create_subwindows(
                windows[i, :, j],
                signal_subwindow_len=duration_in_sec,
                signal_name=signal,
                fs=fs,
            )
            _, yfs = fft_subwindows(subwindows, duration_in_sec, fs=fs)
            padded_yfs = pad_along_axis(yfs, target_length=210, axis=1)
            yfs_averages.append(average_window(padded_yfs)[:210])

        X.append(yfs_averages)
    return np.array(X)


def create_preprocessed_subjects_data(
    subjects_data: dict, fs: int = 64, use_sliding_windows: bool = False
) -> dict:
    # Creates averaged windows for all subjects from dataframes

    subjects_preprosessed_data = {}
    for subject_name, subject_df in subjects_data.items():
        subjects_preprosessed_data[subject_name] = {}
        if use_sliding_windows:
            windows, labels = sliding_windows(subject_df)
        else:
            windows, labels = create_windows(subject_df, fs=fs)

        X = create_training_data_per_subject(fs, windows)
        y = np.array((labels[: len(windows) - 1]))

        subjects_preprosessed_data[subject_name]["X"] = X
        subjects_preprosessed_data[subject_name]["y"] = y

    return subjects_preprosessed_data


def create_training_data_per_subject_gen(fs, windows):
    X = []
    for i in range(0, len(windows) - 1):
        yfs_averages = []
        for j, signal in enumerate(constants.SIGNAL_SUBWINDOW_DICT.keys()):
            duration_in_sec = constants.SIGNAL_SUBWINDOW_DICT[signal]
            subwindows = create_subwindows(
                windows[i, :, j],
                signal_subwindow_len=duration_in_sec,
                signal_name=signal,
                fs=fs,
            )
            _, yfs = fft_subwindows(subwindows, duration_in_sec, fs=fs)

            padded_yfs = pad_along_axis(yfs, target_length=210, axis=1)
            yfs_averages.append(average_window(padded_yfs)[:210])

        X.append(yfs_averages)
    return np.array(X)


def create_preprocessed_subjects_data_gen(windows: np.array, fs: int = 1) -> Tuple:
    # Creates averaged windows for all subjects from dataframes
    print("Windows Shape: ", windows.shape)
    X = create_training_data_per_subject_gen(fs, windows)
    y = np.array((windows[:, 0, 6][: len(X)]))

    return X, y


def get_subject_window_data(subjects_preprosessed_data: Dict) -> Tuple[List, List]:
    # Created train and test data for leave one out cross validation
    all_subjects_X = [
        subject_data["X"] for subject_data in subjects_preprosessed_data.values()
    ]
    all_subjects_y = [
        subject_data["y"] for subject_data in subjects_preprosessed_data.values()
    ]

    return all_subjects_X, all_subjects_y


def save_subject_data(subjects_data, save_path: str):
    # save dictionary as pickle file
    with open(save_path, "wb") as f:
        pickle.dump(subjects_data, f)


def load_data(
    data_type: DataType,
    sampling_rate: int,
    synthetic_data_path: Optional[str] = None,
    use_sliding_windows: bool = False,
) -> Tuple[List, List]:
    """
    Load preprocessed data from disk or create it from scratch.

    Args:
        data_type: The type of data to load (real or synthetic).
        sampling_rate: The sampling rate of the data.
        data_path: The path to the data directory.
        subject_ids: The list of subject IDs to include in the data.
        synthetic_data_path: The path to the synthetic data CSV file.
        load_from_disk: Whether to load data from disk or create it from scratch.

    Returns:
        A tuple of NumPy arrays containing the windowed data and corresponding labels.

    Raises:
        FileNotFoundError: If the data file is not found.
    """

    try:
        # load dictionary from pickle file
        print("*** Try to load original data from disk ***\n")
        with open(f"data/wesad/subject_data_{sampling_rate}hz.pickle", "rb") as f:
            subjects_data = pickle.load(f)
            print("original data loaded")
    except FileNotFoundError:
        print("*** File not found ***")
        print("*** Preprocess data ***")
        dataset = WESADDataset(DATA_PATH, constants.SUBJECT_IDS)
        subjects_data = dataset.get_subject_dataframes(sampling_rate=sampling_rate)
        save_subject_data(
            subjects_data, f"data/wesad/subject_data_{sampling_rate}hz.pickle"
        )

    # if data_type == DataType.DGAN:
    #     print("*** Add synthetic DGAN data ***")
    #     syn_df = pd.read_csv(synthetic_data_path, index_col=0)
    #     subjects_data["SYN"] = syn_df
    #     print(syn_df)

    subjects_preprocessed_data = create_preprocessed_subjects_data(
        subjects_data, fs=sampling_rate, use_sliding_windows=use_sliding_windows
    )
    all_subjects_X, all_subjects_y = get_subject_window_data(subjects_preprocessed_data)

    if data_type == DataType.CGAN:
        print("*** Add synthetic cGAN data ***")
        with open(synthetic_data_path, "rb") as f:
            gen_data = np.load(f)
        X, y = create_preprocessed_subjects_data_gen(gen_data, fs=1)

        all_subjects_X.append(X)
        all_subjects_y.append(y)

    if data_type == DataType.DPCGAN:
        print("*** Add synthetic DP-cGAN data ***")
        with open(synthetic_data_path, "rb") as f:
            gen_data = np.load(f)
        X, y = create_preprocessed_subjects_data_gen(gen_data, fs=1)

        all_subjects_X.append(X)
        all_subjects_y.append(y)

    if data_type == DataType.DGAN:
        print("*** Add synthetic DGAN data ***")
        with open(synthetic_data_path, "rb") as f:
            gen_data = np.load(f)
        X, y = create_preprocessed_subjects_data_gen(gen_data, fs=1)

        all_subjects_X.append(X)
        all_subjects_y.append(y)

    if data_type == DataType.TIMEGAN:
        print("*** Add synthetic TIMEGAN data ***")
        with open(synthetic_data_path, "rb") as f:
            print(f"TimeGAN PATH: {synthetic_data_path}")
            gen_data = np.load(f)

        X, y = create_preprocessed_subjects_data_gen(gen_data, fs=1)

        all_subjects_X.append(X)
        all_subjects_y.append(y)

    return all_subjects_X, all_subjects_y
