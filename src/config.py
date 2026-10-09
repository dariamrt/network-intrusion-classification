from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data_in"
DATA_OUT = ROOT / "data_out"
TABLES = DATA_OUT / "tables"
PLOTS = DATA_OUT / "plots"
TRAIN_FILE = DATA_IN / "Train_data.csv"
APPLY_FILE = DATA_IN / "Test_data.csv"
UNSW_FOLDER = DATA_IN / "unsw_nb15"
APP_RESULTS = DATA_OUT / "app_results.joblib"


def initialize_output_folders():
    for folder in [DATA_OUT, PLOTS, TABLES]:
        folder.mkdir(exist_ok=True)
    for folder in [PLOTS, TABLES]:
        for file in folder.iterdir():
            file.unlink()
