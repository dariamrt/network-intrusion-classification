from pathlib import Path

import joblib

from anomaly_detection import run_anomaly_detection
from cross_dataset import run_cross_dataset
from pipeline import run_pipeline
from robustness import run_robustness

APP_RESULTS = Path("data_out/app_results.joblib")


def build_results(verbose=True):
    results = {
        "pipeline": run_pipeline(verbose),
        "robustness": run_robustness(verbose),
        "cross_dataset": run_cross_dataset(verbose),
        "anomaly_detection": run_anomaly_detection(verbose),
    }
    joblib.dump(results, APP_RESULTS, compress=3)
    return results


def load_results():
    if APP_RESULTS.exists():
        return joblib.load(APP_RESULTS)
    return build_results(verbose=False)


if __name__ == "__main__":
    build_results()
