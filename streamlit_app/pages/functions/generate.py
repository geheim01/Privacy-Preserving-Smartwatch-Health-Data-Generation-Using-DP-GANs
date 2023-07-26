"""
    Auxiliary functions for the synthetic data generation
"""
# passar o datatype para outro sítio??
import pandas as pd
from pages.functions.train import DataType
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# from ydata_synthetic.synthesizers.regular import RegularSynthesizer
# from ydata_synthetic.synthesizers.timeseries import TimeGAN

# def load_model(input_path: str, datatype: DataType):
#     if datatype == DataType.TABULAR:
#         pass
#         #model = RegularSynthesizer.load(input_path)
#     else:
#         pass
#         #model = TimeGAN.load(input_path)
#     #return model


def generate_profile(df: pd.DataFrame):
    report = ProfileReport(df, title="Synthetic data profile", interactions=None)
    st_profile_report(report)
