import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def eval_clustering(Z, labels):
    Z = np.asarray(Z)
    labels = np.asarray(labels)
    return {
        "silhouette": float(silhouette_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
    }


def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    purity_sum = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        _, counts = np.unique(y_true[idx], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / n


def eval_clustering_extended(X, labels, y_true=None):
    X = np.asarray(X)
    labels = np.asarray(labels)
    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }
    if y_true is not None:
        y_true = np.asarray(y_true)
        out.update(
            {
                "ari": float(adjusted_rand_score(y_true, labels)),
                "nmi": float(normalized_mutual_info_score(y_true, labels)),
                "purity": float(purity_score(y_true, labels)),
            }
        )
    return out
