from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def data_split(data, n_split=0.8):
    n = int(n_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# Helper functions for evaluation of the GAN training
# Generate Synthetic Data
def synthetic_dataset(stressdata, nostressdata):
    labels = np.ones((stressdata.shape[0], 1))

    labels_nomos = np.zeros((nostressdata.shape[0], 1))

    trainX_classifier_synthetic = np.concatenate([stressdata, nostressdata], axis=0)
    trainy_classifier_synthetic = np.concatenate([labels, labels_nomos], axis=0)
    return trainX_classifier_synthetic, trainy_classifier_synthetic


def generate_and_plot_data(
    syn_stress: Optional[np.ndarray] = None,
    syn_no_stress: Optional[np.ndarray] = None,
    ori_stress: Optional[np.ndarray] = None,
    ori_no_stress: Optional[np.ndarray] = None,
    samples: int = 0,
    seq_length: int = 0,
) -> "go.Figure":
    """
    Generates and plots synthetic and original stress/no-stress data.

    Each parameter is expected to be a 3D numpy array with shape (samples, seq_length, features).

    Parameters:
    syn_stress: Synthetic stress data.
    syn_no_stress: Synthetic no-stress data.
    ori_stress: Original stress data.
    ori_no_stress: Original no-stress data.
    samples: The number of samples to plot.
    seq_length: The sequence length of the data.

    Returns:
    A plotly figure.
    """

    def add_trace(fig, data, row, col):
        if data is not None:
            fig.add_trace(
                go.Scatter(y=data[i, :, feature], mode="lines"), row=row, col=col
            )

    fig = make_subplots(
        rows=6,
        cols=2,
        subplot_titles=(
            "BVP_SYN_MOS",
            "BVP_REAL_MOS",
            "BVP_SYN_No_MOS",
            "BVP_REAL_No_MOS",
            "EDA_SYN_MOS",
            "EDA_REAL_MOS",
            "EDA_SYN_No_MOS",
            "EDA_REAL_No_MOS",
            "TEMP_SYN_MOS",
            "TEMP_REAL_MOS",
            "TEMP_SYN_No_MOS",
            "TEMP_REAL_No_MOS",
        ),
    )

    for i in range(samples):
        for feature, start_row in zip([0, 1, 5], [1, 3, 5]):
            add_trace(fig, syn_stress, start_row, 1)
            add_trace(fig, ori_stress, start_row, 2)
            add_trace(fig, syn_no_stress, start_row + 1, 1)
            add_trace(fig, ori_no_stress, start_row + 1, 2)

    return fig


def get_optimizer(lr=1e-3, optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.legacy.Adam(
            learning_rate=lr
        )  #  tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.1)


# Build a dataset for Classifier Two sample Test (C2ST)
def build_classifier_dataset(
    synthMos,
    synthNoMos,
    realMos,
    realNoMos,
    synthTestMos,
    synthTestNoMos,
    realTestMos,
    realTestNoMos,
    seq_length,
    num_features,
):
    nn_train_mos = np.concatenate(
        [
            np.reshape(
                realMos, (realMos.shape[0], seq_length * num_features), order="F"
            ),
            np.reshape(
                synthMos, (synthMos.shape[0], seq_length * num_features), order="F"
            ),
        ]
    )
    nn_label_mos = np.concatenate(
        [np.zeros((realMos.shape[0])), np.ones((synthMos.shape[0]))]
    )

    nn_train_nomos = np.concatenate(
        [
            np.reshape(
                realNoMos, (realNoMos.shape[0], seq_length * num_features), order="F"
            ),
            np.reshape(
                synthNoMos, (synthNoMos.shape[0], seq_length * num_features), order="F"
            ),
        ]
    )
    nn_label_nomos = np.concatenate(
        [np.zeros((realNoMos.shape[0])), np.ones((synthNoMos.shape[0]))]
    )

    nn_test_mos = np.concatenate(
        [
            np.reshape(
                realTestMos,
                (realTestMos.shape[0], seq_length * num_features),
                order="F",
            ),
            np.reshape(
                synthTestMos,
                (synthTestMos.shape[0], seq_length * num_features),
                order="F",
            ),
        ]
    )
    nn_label_test_mos = np.concatenate(
        [np.zeros((realTestMos.shape[0])), np.ones((synthTestMos.shape[0]))]
    )

    nn_test_nomos = np.concatenate(
        [
            np.reshape(
                realTestNoMos,
                (realTestNoMos.shape[0], seq_length * num_features),
                order="F",
            ),
            np.reshape(
                synthTestNoMos,
                (synthTestNoMos.shape[0], seq_length * num_features),
                order="F",
            ),
        ]
    )
    nn_label_test_nomos = np.concatenate(
        [np.zeros((realTestNoMos.shape[0])), np.ones((synthTestNoMos.shape[0]))]
    )

    nn_train = np.concatenate([nn_train_mos, nn_train_nomos])
    nn_label = np.concatenate([nn_label_mos, nn_label_nomos])

    nn_test = np.concatenate([nn_test_mos, nn_test_nomos])
    nn_label_test = np.concatenate([nn_label_test_mos, nn_label_test_nomos])

    return nn_train, nn_label, nn_test, nn_label_test
