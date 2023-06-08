# Privacy-Preserving Smartwatch Health Data Generation Using DP-GANs

This repository contains the code and documentation for a research project on generating synthetic smartwatch health data while preserving the privacy of the original data owners. The project uses a combination of differential privacy and generative adversarial networks (DP-GANs) to create synthetic data that closely resembles the original data in terms of statistical properties and data distributions.

![AI generated smartwatch image](images/smartwatch.png)


## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
<!--- - [Acknowledgement](#acknowledgement) --->
<!--- - [License](#license) --->

## Background

Smartwatch health data has become an increasingly popular source of information for healthcare research and personalized medicine. However, the use of such data raises concerns about privacy, as the data often contains sensitive information about individuals' health and fitness. In this project, we aim to address these privacy concerns by generating synthetic health data that can be used in research and analysis while protecting the privacy of the original data owners.

Our approach uses a combination of differential privacy and generative adversarial networks (GANs).

## Requirements

The following are required to run the code in this repository:

- [WESAD Dataset (2,1GB)](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download)
- Python 3.8 or higher
- TensorFlow 2.0 or higher
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Sklearn
- Seaborn
- ...

## Installation

To install the required dependencies, run the following command:

```bash
conda env create -f environment.yml
```

## Usage

The repository consists of multiple notebooks representing the workflow of this work. Every notebook is one step of this workflow starting with the data preprocessing going over to the model training, synthezsing of the new synthesized dataset, to evaluating it with a newly trained respective stress detection model.

**01-Data**

The data is loaded from the original WESAD dataset preprocessed and saved within a new file under a new named file [wesad_preprocessed_1hz.csv](data/wesad/wesad_preprocessed_1hz.csv).

**[02-Model](02-Model.ipynb)**

This notebook focuses on training the CGAN model. It loads the preprocessed data from the previous 01-Data notebook and transforms it into multiple time-related windows for training.

**[03-Generator](03-Generator.ipynb)**

The generator notebook is responsible for synthesizing a new dataset based on the trained GAN model. The generated data is saved separately in the [syn data folder](data/syn/cgan).

**[04-Evaluation](04-Evaluation.ipynb)**

In the evaluation notebook, we assess the quality of the synthetically generated dataset using visual and statistical metrics. The usefullness evaluation takes place in the [05-Stress_Detection](05-Stress_Detection.ipynb) notebook.

**[05-Stress_Detection](05-Stress_Detection.ipynb)**

This notebook focuses on training a CNN model to perform stress detection on the synthetic dataset, simulating a real-world use case.

<!--- ## Acknowledgement --->

<!--- ## License --->