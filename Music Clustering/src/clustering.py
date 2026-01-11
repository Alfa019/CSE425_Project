import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_reduce(Z, n_components=8, seed=42):
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(Z), pca


def tune_kmeans(Z, k_list, seed=42, n_init=50):
    from evaluation import eval_clustering

    rows = []
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        y = km.fit_predict(Z)
        m = eval_clustering(Z, y)
        m.update({"K": k})
        rows.append(m)
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


def cluster_kmeans(Z, k, seed=42, n_init=50):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(Z)
    return labels, km


def tsne_embed(Z, seed=42, perplexity=35):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    return tsne.fit_transform(Z)


def plot_tsne(Z, labels, title):
    try:
        import plotly.express as px
    except Exception as exc:
        raise RuntimeError("plotly is required for plot_tsne().") from exc

    Z2 = tsne_embed(Z)
    fig = px.scatter(
        x=Z2[:, 0],
        y=Z2[:, 1],
        color=np.asarray(labels).astype(str),
        title=title,
        labels={"x": "t-SNE-1", "y": "t-SNE-2", "color": "cluster"},
    )
    fig.show()
    return Z2


def evaluate_clustering_with_labels(X, labels, y_true=None, name="method"):
    from evaluation import eval_clustering_extended

    X = np.asarray(X)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    n_clusters = len(uniq[uniq != -1]) if (-1 in uniq) else len(uniq)

    row = {"method": name, "n_clusters": int(n_clusters)}
    try:
        if n_clusters >= 2:
            if -1 in uniq:
                mask = labels != -1
                metrics = eval_clustering_extended(X[mask], labels[mask], None if y_true is None else np.asarray(y_true)[mask])
                row["noise_frac"] = float(np.mean(~mask))
            else:
                metrics = eval_clustering_extended(X, labels, y_true)
                row["noise_frac"] = 0.0
            row.update(metrics)
        else:
            row.update(
                {
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "ari": np.nan if y_true is not None else None,
                    "nmi": np.nan if y_true is not None else None,
                    "purity": np.nan if y_true is not None else None,
                    "noise_frac": float(np.mean(labels == -1)) if (-1 in uniq) else 0.0,
                }
            )
    except Exception:
        row.update(
            {
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "ari": np.nan if y_true is not None else None,
                "nmi": np.nan if y_true is not None else None,
                "purity": np.nan if y_true is not None else None,
                "noise_frac": float(np.mean(labels == -1)) if (-1 in uniq) else 0.0,
            }
        )
    return row


def run_clusterings(X, y_true, k, seed=42):
    rows = []

    km_labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(X)
    rows.append(evaluate_clustering_with_labels(X, km_labels, y_true, "KMeans"))

    agg_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    rows.append(evaluate_clustering_with_labels(X, agg_labels, y_true, "Agglomerative"))

    db_labels = DBSCAN(eps=1.5, min_samples=10, metric="euclidean").fit_predict(X)
    rows.append(evaluate_clustering_with_labels(X, db_labels, y_true, "DBSCAN"))

    return pd.DataFrame(rows)
