import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def silverman_bandwidth(x):
    n = len(x)
    sigma = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    h = 0.9 * min(sigma, iqr / 1.34) * n ** (-1 / 5)
    if h == 0:
        h = 0.05
    if np.isnan(h):
        h = 1.0
    return h


def estimate_h_KDE(x, compute_sigma=False):
    if compute_sigma:
        sigma = np.std(x, axis=0).mean()
        params = {'bandwidth': np.linspace(0.1 * sigma, 2 * sigma, 30)}
    else:
        params = {'bandwidth': np.linspace(0.1, 2, 30)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
    grid.fit(x)
    return grid.best_params_["bandwidth"]


class KDENaiveBayes:
    def __init__(self, h=None, kernel='gaussian'):
        self.h = h
        self.kernel = kernel
        self.classes_ = None
        self.models = {}
        self.log_prior_prob = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n, d = X.shape
        for c in self.classes_:
            Xc = X[y == c]
            self.log_prior_prob[c] = np.log(len(Xc) / n)
            self.models[c] = []
            for j in range(d):
                if self.h is None:
                    h = silverman_bandwidth(Xc[:, j])
                    kde = KernelDensity(bandwidth=h, kernel=self.kernel)
                else:
                    kde = KernelDensity(bandwidth=self.h, kernel=self.kernel)
                kde.fit(Xc[:, j].reshape(-1, 1))
                self.models[c].append(kde)

    def predict_proba(self, x):
        n, d = x.shape
        log_scores = np.zeros((n, len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            log_score = np.full(n, self.log_prior_prob[c])
            for j in range(d):
                kde = self.models[c][j]
                log_score += kde.score_samples(x[:, j].reshape(-1, 1))
            log_scores[:, idx] = log_score
        max_log = np.max(log_scores, axis=1, keepdims=True)
        probabilities = np.exp(log_scores - max_log)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]
