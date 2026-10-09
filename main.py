import joblib

from src.analysis.anomaly_detection import run_anomaly_detection
from src.analysis.cross_dataset import run_cross_dataset
from src.analysis.robustness import run_robustness
from src.config import APP_RESULTS
from src.pipeline import run_pipeline


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
