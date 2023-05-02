import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy

from synthesizers.utils.preprocessing import get_max_value_from_list


class Subject:
    """Subject of the WESAD dataset.
    Subject Class inspired by: https://github.com/WJMatthew/WESAD"""

    def __init__(self, main_path, subject_number):
        self.name = f"S{subject_number}"
        self.subject_keys = ["signal", "label", "subject"]
        self.signal_keys = ["chest", "wrist"]
        self.chest_keys = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]
        self.wrist_keys = ["ACC", "BVP", "EDA", "TEMP"]
        with open(os.path.join(main_path, self.name, f"{self.name}.pkl"), "rb") as file:
            self.data = pickle.load(file, encoding="latin1")
        self.labels = self.data["label"]

    def get_wrist_data(self):
        """Returns data measured by the E4 Empatica"""

        data = self.data["signal"]["wrist"]
        return data

    def get_subject_dataframe(self, sampling_rate: int = 64):
        """Returns a dataframe with the preprocessed data of the subject"""
        wrist_data = self.get_wrist_data()
        bvp_signal = wrist_data["BVP"][:, 0]
        eda_signal = wrist_data["EDA"][:, 0]
        acc_x_signal = wrist_data["ACC"][:, 0]
        acc_y_signal = wrist_data["ACC"][:, 1]
        acc_z_signal = wrist_data["ACC"][:, 2]
        temp_signal = wrist_data["TEMP"][:, 0]
        # Upsampling data to match BVP data sampling rate using fourier method as described in Paper/dataset
        num_samples = (len(self.labels) // 700) * sampling_rate
        bvp_resampled = scipy.signal.resample(bvp_signal, num_samples)
        eda_resampled = scipy.signal.resample(eda_signal, num_samples)
        temp_resampled = scipy.signal.resample(temp_signal, num_samples)
        acc_x_resampled = scipy.signal.resample(acc_x_signal, num_samples)
        acc_y_resampled = scipy.signal.resample(acc_y_signal, num_samples)
        acc_z_resampled = scipy.signal.resample(acc_z_signal, num_samples)
        label_df = pd.DataFrame(self.labels, columns=["label"])
        label_df.index = [
            (1 / 700) * i for i in range(len(label_df))
        ]  # 700 is the sampling rate of the label
        label_df.index = pd.to_datetime(label_df.index, unit="s")
        data_arrays = zip(
            bvp_resampled,
            eda_resampled,
            acc_x_resampled,
            acc_y_resampled,
            acc_z_resampled,
            temp_resampled,
        )
        df = pd.DataFrame(
            data=data_arrays, columns=["BVP", "EDA", "ACC_x", "ACC_y", "ACC_z", "TEMP"]
        )
        df.index = [
            (1 / sampling_rate) * i for i in range(len(df))
        ]  # 64 = sampling rate of BVP
        df.index = pd.to_datetime(df.index, unit="s")
        df = df.join(label_df)
        df["label"] = df["label"].fillna(method="ffill")
        df.reset_index(drop=True, inplace=True)
        df.drop(df[df["label"].isin([0.0, 4.0, 5.0, 6.0, 7.0])].index, inplace=True)
        df["label"] = df["label"].replace([1.0, 2.0, 3.0], [0, 1, 0])
        df.reset_index(drop=True, inplace=True)
        df = (df - df.min()) / (
            df.max() - df.min()
        )  # Normalize data (no train test leakage since data frame per subject)
        return df


class WESADDataset:
    """Wrapper class for multiple Subject instances from the WESAD dataset."""

    def __init__(self, main_path: str, subject_numbers: List[int]):
        self.subjects = {}
        for num in subject_numbers:
            self.subjects[num] = Subject(main_path, num)

    def get_subject_dataframes(self, sampling_rate: int = 64):
        """Returns a dictionary of preprocessed dataframes for each subject."""
        dataframes = {}
        for num, subject in self.subjects.items():
            dataframes[num] = subject.get_subject_dataframe(sampling_rate=sampling_rate)
        return dataframes

    def get_all_data(self, sampling_rate: int = 64):
        """Returns a concatenated dataframe of preprocessed data from all subjects."""
        dfs = self.get_subject_dataframes(sampling_rate=sampling_rate)
        df = pd.concat(dfs.values())
        df = df.reset_index(drop=True)
        return df

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
            label = int(
                get_max_value_from_list(df["label"][i : i + window_len].to_list())
            )

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
