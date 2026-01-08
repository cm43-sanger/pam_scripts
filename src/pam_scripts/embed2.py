import argparse
import sys
import typing
import warnings
import numpy as np
import umap
from collections import defaultdict
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from . import _core, pam_io

DEFAULT_THRESHOLD = 0.001
DEFAULT_EPS = 0.05
DEFAULT_MIN_SAMPLES = 5


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


def embed(
    distances,
    normalize: bool = True,
    seed: typing.Optional[int] = None,
    num_jobs: int = _core.NUM_CPUS,
):
    distances = np.asarray(distances, dtype=np.float64)
    reducer = umap.UMAP(metric="precomputed", n_jobs=num_jobs, random_state=seed)
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


def cluster_embedding(
    z,
    hierarchical: bool = False,
    eps: typing.Optional[float] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    num_jobs: int = _core.NUM_CPUS,
):
    if not hierarchical:
        eps = eps or DEFAULT_EPS
        return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=num_jobs).fit(z)
    if eps:
        print(f"eps={eps} ignored by HDBSCAN", file=sys.stderr)
    return HDBSCAN(min_samples=min_samples, core_dist_n_jobs=num_jobs).fit(z)


def embed_distances(
    input_phylip: str,
    threshold: float = DEFAULT_THRESHOLD,
    seed: typing.Optional[int] = None,
    hierarchical: bool = False,
    eps: typing.Optional[float] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    num_jobs: int = _core.NUM_CPUS,
):
    names, distances = pam_io.load_distance_matrix(input_phylip)
    z = embed(distances, seed=seed, num_jobs=num_jobs)
    clusters = cluster_embedding(
        z,
        hierarchical=hierarchical,
        eps=eps,
        min_samples=min_samples,
        num_jobs=num_jobs,
    )
    density_labels = clusters.labels_
    labels = ["unclustered"] * len(names)
    labels2 = ["unclustered"] * len(names)
    for density_label in range(density_labels.max() + 1):
        (indices,) = np.nonzero(density_labels == density_label)
        if indices.size == 0:
            continue
        cluster_distances = distances[np.ix_(indices, indices)]
        Z = linkage(squareform(cluster_distances), method="complete")
        sub_labels = fcluster(Z, t=threshold, criterion="distance")
        sub_clusters = defaultdict(list)
        for i, sub_label in enumerate(sub_labels):
            sub_clusters[sub_label].append(i)
        for sub_label, idx in enumerate(sub_clusters.values()):
            ooer = f"{density_label + 1}.{sub_label + 1}"
            with open(f"{ooer}.txt", "w") as f:
                for i in idx:
                    print(names[indices[i]], file=f)
                    labels[indices[i]] = density_label + 1
                    labels2[indices[i]] = sub_label + 1
    return (names, z, labels, labels2)


def write_embedding(
    filename: str, names: list[str], z, labels: list[str], sub_labels: list[str]
):
    with pam_io.get_output_handle(filename) as f:
        print("name", "x", "y", "label", "sub_label", sep="\t", file=f)
        for name, (x, y), label, sub_label in zip(names, z, labels, sub_labels):
            print(name, x, y, label, sub_label, sep="\t", file=f)


def main():
    parser = argparse.ArgumentParser(
        description="Embed PHYLIP distance matrix using UMAP and cluster with DBSCAN."
    )
    parser.add_argument(
        "input_phylip", nargs="?", default="-", help="Input PHYLIP file"
    )
    parser.add_argument(
        "--output_tsv", "-o", required=True, help="Output TSV file (defaults to stdout)"
    )
    # parser.add_argument("--output_groups", "-g", required=True, help="Output groups")
    parser.add_argument(
        "-d",
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for UMAP "
        "(default None, WARNING: renders UMAP single-threaded if set)",
    )
    parser.add_argument(
        "-H",
        "--hierarchical",
        action="store_true",
        help="Use hierarchical clustering (HDBSCAN)",
    )
    parser.add_argument(
        "--num_jobs",
        "-t",
        type=int,
        default=_core.NUM_CPUS,
        help=f"Number of jobs for embedding and clustering (default: {_core.NUM_CPUS})",
    )
    parser.add_argument(
        "--threshold",
        "-c",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold cluster (default: {DEFAULT_THRESHOLD})",
    )
    dbscan_group = parser.add_argument_group("(H)DBSCAN clustering options")
    dbscan_group.add_argument(
        "--eps",
        "-e",
        type=float,
        default=None,
        help="Maximum distance between neighbouring samples in cluster"
        f"(default: {DEFAULT_EPS})",
    )
    dbscan_group.add_argument(
        "--min_samples",
        "-m",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Minimum samples per cluster (default: {DEFAULT_MIN_SAMPLES})",
    )
    args = parser.parse_args()

    if args.hierarchical:
        raise NotImplementedError("no HDBSCAN pls")

    names, z, clusters, sub_clusters = embed_distances(
        args.input_phylip,
        threshold=args.threshold,
        seed=args.seed,
        hierarchical=args.hierarchical,
        eps=args.eps,
        min_samples=args.min_samples,
        num_jobs=args.num_jobs,
    )
    write_embedding(args.output_tsv, names, z, clusters, sub_clusters)


if __name__ == "__main__":
    main()
