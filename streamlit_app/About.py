"""
    smartwatch-synthy streamlit app landing page
    inspired by the y-data frontend for synthetic data generation: https://github.com/ydataai/ydata-synthetic
"""
import streamlit as st


def create_landing_page():
    col1, col2 = st.columns([2, 4])

    with col1:
        st.image("images/smartwatch.png", width=200)

    with col2:
        st.title("Welcome to Smartwatch Synthy!")
        st.text("Privacy Preserving Smartwatch Health Data Generation using DPGANs")

    st.markdown("The following are the comprised results of my master thesis.")
    create_synthetic_data_section()
    st.markdown(
        "This *streamlit_app* application can generate synthetic data for the WESAD dataset. "
    )
    # create_training_section()
    create_generation_section()


def create_synthetic_data_section():
    st.header("What is synthetic data?")
    st.markdown(
        "Synthetic data is artificially generated data that is not collected from real-world events. "
        "It replicates the statistical components of real data containing no identifiable information, "
        "ensuring an individual’s privacy."
    )

    st.header("Why Synthetic Data?")
    st.markdown(
        "Synthetic data can be used for many applications: \n"
        "- Privacy\n"
        "- Remove bias\n"
        "- Balance datasets\n"
        "- Augment datasets"
    )


# def create_training_section():
#     st.subheader("Select & train a synthesizer")
#     st.markdown(
#         "`Smartwatch-Synth` streamlit app enables the training and generation of synthetic data from "
#         "generative architectures. The current app only provides support for the generation tabular data "
#         "and for the following architectures:\n"
#         "- GAN\n"
#         "- CGAN\n"
#         "- TimeGAN\n"
#     )

#     st.markdown(
#         "##### What you should ensure before training the synthesizer:\n"
#         "- Make sure your dataset has no missing data. If missing data is a problem, no worries. "
#         "Check the article and this article. \n"
#         "- Make sure you choose the right number of epochs and batch_size considering your dataset shape. "
#         "The choice of these 2 parameters highly affects the results you may get. \n"
#         "- Make sure that you've the right data types selected. Only numerical and categorical values "
#         "are supported. In case date, datetime, or text is available in the dataset, the columns should "
#         "be preprocessed before the model training."
#     )

#     st.markdown(
#         "The trained synthesizer is saved to `*.trained_synth.pkl*` by default."
#     )


def create_generation_section():
    st.subheader("Generate & compare synthetic samples")

    st.markdown(
        "The smartwatch-synthy app experience allows you to:\n"
        "- Generate as many samples as you want based on the provided input\n"
        "- Generate a profile for the generated synthetic samples\n"
        "- Save the generated samples to a local directory"
    )

    st.markdown(
        "##### What you should ensure before generating synthetic samples:\n"
        "- The models are found in `/models/`.\n"
        "- To add new models, extend the model_dict in `2_Generate_synthetic_data.py`"
    )


def main():
    st.set_page_config(
        page_title="Smartwatch Synthy - Synthetic WESAD dataset generation streamlit_app",
        page_icon="👋",
        layout="wide",
    )
    create_landing_page()


if __name__ == "__main__":
    main()
