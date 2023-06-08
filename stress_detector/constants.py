# Subwindow length of the biosignals
SIGNAL_SUBWINDOW_DICT = {
    "ACC_x": 7,
    "ACC_y": 7,
    "ACC_z": 7,
    "BVP": 30,
    "EDA": 30,
    "TEMP": 35,
}

SMARTWATCH_OS = {
    "E4": ["ACC_x", "ACC_y", "ACC_z", "TEMP", "EDA", "BVP"],
    "E4_DGAN": ["ACC_x", "ACC_y", "ACC_z", "TEMP", "EDA", "BVP"],
    "E4_CGAN": ["ACC_x", "ACC_y", "ACC_z", "TEMP", "EDA", "BVP"],
    #'Tizen': ['ACC_x', 'ACC_y', 'ACC_z', 'TEMP', 'BVP'],
    #'WearOS_watchOS': ['ACC_x', 'ACC_y', 'ACC_z', 'TEMP'],
    #'Fitbit': ['ACC_x', 'ACC_y', 'ACC_z', 'TEMP', 'EDA'],
    #'PiaOS': ['TEMP', 'EDA', 'BVP']
}

SIGNALS = ["ACC_x", "ACC_y", "ACC_z", "TEMP", "EDA", "BVP"]

SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

DATA_PATH = "/Users/nils/thesis/Data_Generation/data/wesad/wesad_preprocessed_1hz.csv"
