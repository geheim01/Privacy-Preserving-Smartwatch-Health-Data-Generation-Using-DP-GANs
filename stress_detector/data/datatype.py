from enum import Enum


class DataType(Enum):
    CGAN = 0
    CGAN_LSTM = 1
    CGAN_FCN = 2
    CGAN_TRANSFORMER = 3
    DPCGAN = 4
    DGAN = 5
    TIMEGAN = 6
    REAL = 7
