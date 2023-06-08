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
def generate_and_plot_data(
    syn_stress, syn_no_stress, ori_stress, ori_no_stress, samples, seq_length
):
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
        # BVP
        fig.add_trace(go.Scatter(y=syn_stress[i, :, 0], mode="lines"), row=1, col=1)

        fig.add_trace(go.Scatter(y=ori_stress[i, :, 0], mode="lines"), row=1, col=2)

        fig.add_trace(go.Scatter(y=syn_no_stress[i, :, 0], mode="lines"), row=2, col=1)

        fig.add_trace(go.Scatter(y=ori_no_stress[i, :, 0], mode="lines"), row=2, col=2)

        # EDA
        fig.add_trace(go.Scatter(y=syn_stress[i, :, 1], mode="lines"), row=3, col=1)

        fig.add_trace(go.Scatter(y=ori_stress[i, :, 1], mode="lines"), row=3, col=2)

        fig.add_trace(go.Scatter(y=syn_no_stress[i, :, 1], mode="lines"), row=4, col=1)

        fig.add_trace(go.Scatter(y=ori_no_stress[i, :, 1], mode="lines"), row=4, col=2)

        # TEMP
        fig.add_trace(go.Scatter(y=syn_stress[i, :, 5], mode="lines"), row=5, col=1)

        fig.add_trace(go.Scatter(y=ori_stress[i, :, 5], mode="lines"), row=5, col=2)

        fig.add_trace(go.Scatter(y=syn_no_stress[i, :, 5], mode="lines"), row=6, col=1)

        fig.add_trace(go.Scatter(y=ori_no_stress[i, :, 5], mode="lines"), row=6, col=2)

    # fig.show()

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
