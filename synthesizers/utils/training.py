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


# Plot the generated data in a 2x2 grid subplot
def generate_and_plot_data(predictions, predictions2, samples, seq_length):
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "BVP_MOS",
            "BVP_NoMOS",
            "EDA_MOS",
            "EDA_NoMOS",
            "TEMP_MOS",
            "TEMP_noMOS",
        ),
    )

    for i in range(samples):
        fig.add_trace(go.Scatter(y=predictions[i, :, 0], mode="lines"), row=1, col=1)

        fig.add_trace(go.Scatter(y=predictions2[i, :, 0], mode="lines"), row=1, col=2)

        fig.add_trace(go.Scatter(y=predictions[i, :, 1], mode="lines"), row=2, col=1)

        fig.add_trace(go.Scatter(y=predictions2[i, :, 1], mode="lines"), row=2, col=2)

        fig.add_trace(go.Scatter(y=predictions[i, :, 5], mode="lines"), row=3, col=1)

        fig.add_trace(go.Scatter(y=predictions2[i, :, 5], mode="lines"), row=3, col=2)

    return fig.show()


# Build a Data set for Classifier Two sample Test
def buildDataCTST(
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
