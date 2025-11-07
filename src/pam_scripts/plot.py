import argparse
import typing
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from . import pam_io

DEFAULT_DPI = 200.0


def bounding_square(x, y, margin=0.02):
    x = np.asarray(x)
    y = np.asarray(y)
    x0 = x.min() - margin
    x1 = x.max() + margin
    y0 = y.min() - margin
    y1 = y.max() + margin
    return ((x0, x1, x1, x0, x0), (y0, y0, y1, y1, y0))


def plot_embedding(
    embedding: pd.DataFrame,
    s: float = 8.0,
    alpha: float = 0.8,
    palette: str = "tab10",
    subplots_kwargs: typing.Optional[dict] = None,
) -> Figure:
    if subplots_kwargs is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = plt.subplots(**subplots_kwargs)
    sns.scatterplot(
        data=embedding,
        x="x",
        y="y",
        hue="label",
        s=s,
        alpha=alpha,
        ax=axis,
        palette=palette,
        legend=False,
    )
    for label in embedding["label"].unique():
        if label == -1:
            continue
        subset = embedding[embedding["label"] == label]
        x, y = bounding_square(subset["x"], subset["y"])
        axis.plot(x, y, "k:", linewidth=1.0)
        # print(label)
        # sns.kdeplot(
        #     x=subset["x"],
        #     y=subset["y"],
        #     ax=axis,
        #     levels=[0.05],
        #     color="k",
        #     linewidths=1.0,
        # )
    axis.set_xlabel("$X$")
    axis.set_ylabel("$Y$")
    axis.axis("equal")
    fig.tight_layout()
    return fig


def normalize_embedding(
    z,
) -> np.ndarray[tuple[int, typing.Literal[2]], np.dtype[np.float64]]:
    z = np.asarray(z, dtype=np.float64)
    pca = PCA(n_components=2).fit(z)
    scale = np.sqrt(pca.explained_variance_.sum())
    if scale < 1e-5:
        warnings.warn("Variance is zero; only centering embedding")
        return z - pca.mean_
    matrix = pca.components_.T / scale
    return (z - pca.mean_) @ matrix


def embed(distances, normalize: bool = True, num_jobs: int = 1):
    distances = np.asarray(distances, dtype=np.float64)
    reducer = umap.UMAP(metric="precomputed", n_jobs=num_jobs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="using precomputed metric; inverse_transform will be unavailable",
            category=UserWarning,
            module="umap.umap_",
        )
        z = reducer.fit_transform(distances)
    if normalize:
        z = normalize_embedding(z)
    return typing.cast(
        np.ndarray[tuple[int, typing.Literal[2]], np.dtype[np.float64]], z
    )  # need to cast or Pylance complains


def cluster_embedding(z, eps: float = 0.05, min_samples: int = 10, num_jobs: int = 1):
    clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=num_jobs)
    return clusters.fit(z)


def embed_distances(
    input_phylip: str, num_jobs: int = 1, eps: float = 0.05, min_samples: int = 10
):
    names, distances = pam_io.load_distance_matrix(input_phylip)
    z = embed(distances, num_jobs=num_jobs)
    clusters = cluster_embedding(z, eps=eps, min_samples=min_samples, num_jobs=num_jobs)
    return (names, z, clusters)


def write_embedding(filename: str, names: list[str], z, clusters: DBSCAN):
    with pam_io.get_output_handle(filename) as f:
        print("name", "x", "y", "label", sep="\t", file=f)
        for name, (x, y), label in zip(names, z, clusters.labels_):
            print(name, x, y, label, sep="\t", file=f)


def main():
    parser = argparse.ArgumentParser(
        description="Embed PHYLIP distance matrix using UMAP and cluster with DBSCAN."
    )
    parser.add_argument(
        "input_tsv", nargs="?", default="-", help="Input tsv file (defaults to stdin)"
    )
    parser.add_argument(
        "output_png",
        nargs="?",
        default="-",
        help="Output png file (defaults to stdout)",
    )
    parser.add_argument(
        "-d",
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help=f"DPI for output PNG (default {DEFAULT_DPI})",
    )
    parser.add_argument(
        "-e",
        "--interactive",
        action="store_true",
        help="Show interactive plot of the embedding",
    )
    args = parser.parse_args()

    with pam_io.get_input_handle(args.input_tsv) as f:
        embedding = pd.read_csv(f, sep="\t")
    fig = plot_embedding(embedding)
    dpi = args.dpi
    try:
        dpi = float(dpi)
        if dpi <= 0.0:
            raise ValueError
    except ValueError:
        raise ValueError(f"invalid DPI value: {dpi!r}")
    with pam_io.get_output_handle(args.output_png, "wb") as f:
        fig.savefig(f, format="png", dpi=dpi)
    if args.interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
