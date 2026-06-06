import pandas as pd
from downloadfile import download_pubg_data

DATA_PATH = "data/Pubg_Stats.csv"

try:
    ori_data = pd.read_csv(DATA_PATH)
    if ori_data is None or ori_data.empty:
        raise ValueError("Empty dataframe")
except (FileNotFoundError, ValueError):
    download_pubg_data()
    ori_data = pd.read_csv(DATA_PATH)

print(ori_data.head())
