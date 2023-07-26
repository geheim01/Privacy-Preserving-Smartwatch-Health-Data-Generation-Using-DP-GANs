"""
PCA and tSNE visualization plot is from Yoo et al. from the following publication:

Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len]
            )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (
                    prep_data_hat,
                    np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                )
            )

    # Visualization parameter
    colors = ["tab:blue" for i in range(anal_sample_no)] + [
        "tab:orange" for i in range(anal_sample_no)
    ]

    if analysis == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")

        return plt

    elif analysis == "tsne":
        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")

        return plt


def plot_signal_distributions(X, syn_data, SIGTOI, ITOSIG):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Compare distribution of T_out values
    for sig_num, ax in zip(range(len(SIGTOI)), axs.flatten()):
        ax.hist(
            [
                np.array(X[:, :, sig_num]).flatten(),
                np.array(syn_data[:, :, sig_num]).flatten(),
            ],
            label=["Real", "Synthetic"],
            bins=25,
            density=True,
        )
        ax.legend()
        ax.set_xlabel(ITOSIG[sig_num])
        ax.set_ylabel("Density")
        ax.set_title("Statistical Distribution - Real vs Synthetic Data")

    plt.tight_layout()
    # plt.show()

    return plt


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


# def generate_and_plot_data(
#     syn_stress: Optional[np.ndarray] = None,
#     syn_no_stress: Optional[np.ndarray] = None,
#     ori_stress: Optional[np.ndarray] = None,
#     ori_no_stress: Optional[np.ndarray] = None,
#     samples: int = 0,
#     seq_length: int = 0,
# ):
#     """
#     Generates and plots synthetic and original stress/no-stress data.

#     Each parameter is expected to be a 3D numpy array with shape (samples, seq_length, features).

#     Parameters:
#     syn_stress: Synthetic stress data.
#     syn_no_stress: Synthetic no-stress data.
#     ori_stress: Original stress data.
#     ori_no_stress: Original no-stress data.
#     samples: The number of samples to plot.
#     seq_length: The sequence length of the data.

#     Returns:
#     A plotly figure.
#     """

#     def add_trace(fig, data, row, col):
#         if data is not None:
#             fig.add_trace(
#                 go.Scatter(y=data[i, :, feature], mode="lines"), row=row, col=col
#             )

#     fig = make_subplots(
#         rows=6,
#         cols=2,
#         subplot_titles=(
#             "BVP_SYN_MOS",
#             "BVP_REAL_MOS",
#             "BVP_SYN_No_MOS",
#             "BVP_REAL_No_MOS",
#             "EDA_SYN_MOS",
#             "EDA_REAL_MOS",
#             "EDA_SYN_No_MOS",
#             "EDA_REAL_No_MOS",
#             "TEMP_SYN_MOS",
#             "TEMP_REAL_MOS",
#             "TEMP_SYN_No_MOS",
#             "TEMP_REAL_No_MOS",
#         ),
#     )

#     for i in range(samples):
#         for feature, start_row in zip([0, 1, 5], [1, 3, 5]):
#             add_trace(fig, syn_stress, start_row, 1)
#             add_trace(fig, ori_stress, start_row, 2)
#             add_trace(fig, syn_no_stress, start_row + 1, 1)
#             add_trace(fig, ori_no_stress, start_row + 1, 2)

#     return fig
