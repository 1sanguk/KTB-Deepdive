import subprocess
import os

def download_pubg_data():
    subprocess.run(["pip", "install", "opendatasets"], check=True)

    import opendatasets as od  # noqa: E402

    os.makedirs("data", exist_ok=True)
    od.download("https://www.kaggle.com/datasets/mohammadtalib786/pubg-stats-dataset", data_dir="data/")
