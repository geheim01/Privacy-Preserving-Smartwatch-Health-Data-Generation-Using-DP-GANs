from enum import Enum

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


class StressType(Enum):
    NON_STRESS = "Non-Stress"
    STRESS = "Stress"
    BOTH = "Both"


MODEL_DICT = {
    "cGAN": "cgan/resilient_sweep-1",
    "DP-cGAN-e-0.1": "dp-cgan-e-0_1/light-sweep-1",
    "DP-cGAN-e-1": "dp-cgan-e-1/revived-sweep-2",
    "DP-cGAN-e-10": "dp-cgan-e-10/usual-sweep-3",
}  # Update with your model names or paths

ITOSIG = {
    0: "BVP",
    1: "EDA",
    2: "ACC_x",
    3: "ACC_y",
    4: "ACC_z",
    5: "TEMP",
    6: "Label",
}


def generate_samples(
    model: keras.models.Model, num_samples: int, latent_dim: int, label_value: int
) -> np.ndarray:
    """
    Generate synthetic data samples.

    Args:
        model: Trained model used to generate the synthetic samples.
        num_samples: The number of synthetic samples to generate.
        latent_dim: The dimension of the latent space in the model.
        label_value: The label value of the synthetic samples to generate.

    Returns:
        A numpy array of the generated synthetic samples.
    """

    # Check that num_samples and latent_dim are positive integers
    assert (
        isinstance(num_samples, int) and num_samples > 0
    ), "num_samples should be a positive integer"
    assert (
        isinstance(latent_dim, int) and latent_dim > 0
    ), "latent_dim should be a positive integer"

    labels = tf.fill([num_samples, 1], label_value)
    append_value = np.full([num_samples, 60, 1], label_value)

    random_vector = tf.random.normal(shape=(num_samples, latent_dim))
    synth_samples = model([random_vector, labels])
    synth_samples = np.append(np.array(synth_samples), append_value, axis=2)

    return synth_samples


def generate(
    model: keras.models.Model,
    num_syn_samples: int,
    latent_dim: int,
    stress_type: StressType,
) -> np.ndarray:
    if not isinstance(stress_type, StressType):
        raise ValueError(
            f"stress_type must be an instance of StressType Enum, got {stress_type}"
        )

    if stress_type in [StressType.NON_STRESS, StressType.STRESS]:
        label_value = 0 if stress_type == StressType.NON_STRESS else 1
        synth_samples = generate_samples(
            model, num_syn_samples, latent_dim, label_value
        )
    else:
        num_samples_half = num_syn_samples // 2
        non_stress_samples = generate_samples(model, num_samples_half, latent_dim, 0)
        stress_samples = generate_samples(model, num_samples_half, latent_dim, 1)
        synth_samples = np.concatenate((non_stress_samples, stress_samples))

    return synth_samples


def plot_generated_data(data: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[0]):
        plt.plot(data[i, :, 0], label=f"Sample {i+1}")
    plt.title(title)
    plt.xlabel("Time in seconds (s)")
    plt.ylabel("Signal Value")
    plt.ylim([0, 1])
    plt.grid(True)
    st.pyplot()


def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def load_model(model_name: str) -> keras.models.Model:
    return keras.models.load_model(f"models/{model_name}/generator")


def generate_synthetic_data(
    model: keras.models.Model, num_syn_samples: int, latent_dim: int, stress_type: str
) -> pd.DataFrame:
    synth_data = generate(model, num_syn_samples, latent_dim, stress_type)
    df = pd.DataFrame(synth_data.reshape(-1, synth_data.shape[-1]))

    # Add column names
    df.columns = [ITOSIG[i] for i in range(synth_data.shape[-1])]

    # Add index
    df.index = pd.RangeIndex(1, len(df) + 1)

    return df


def display_dataframe(df: pd.DataFrame) -> None:
    st.dataframe(df)
    # For each signal in the synthetic data
    st.line_chart(df, height=200)

    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)


def download_dataframe(df: pd.DataFrame, model_name: str) -> None:
    csv = convert_df(df)
    st.download_button(
        "Download synthetic data as CSV",
        csv,
        file_name=f"synthetic_{model_name}.csv",
        mime="text/csv",
        key="down-load-csv",
    )


def run():
    st.subheader("Generate synthetic data from a trained model")
    col1, col2 = st.columns([4, 2])
    with col1:
        model_selection = st.selectbox(
            "Select the model", ["cGAN", "DP-cGAN-e-0.1", "DP-cGAN-e-1", "DP-cGAN-e-10"]
        )
        model_name = MODEL_DICT[model_selection]
        latent_dim = st.number_input(
            "Length of windows in seconds", min_value=0, value=60
        )
        num_syn_samples = st.number_input(
            "Number of synthetic windows to generate", min_value=0, value=10
        )
        stress_type_str = st.selectbox(
            "Select type of data to generate", [e.value for e in StressType]
        )

        # Map string back to StressType Enum
        stress_type = StressType(stress_type_str)

    if st.button("Generate samples"):
        try:
            model = load_model(model_name)
            st.success(
                "The model was properly loaded and is now ready to generate synthetic samples!"
            )
            with st.spinner("Generating samples... This might take time."):
                df = generate_synthetic_data(
                    model, num_syn_samples, latent_dim, stress_type
                )

            display_dataframe(df)
            st.success("Synthetic data has been generated successfully!")
            download_dataframe(df, model_name)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run()
