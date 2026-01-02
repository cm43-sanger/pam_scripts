import argparse
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
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


def compare_embeddings(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    s: float = 8.0,
    alpha: float = 0.8,
    palette: str = "tab10",
    subplots_kwargs: typing.Optional[dict] = None,
) -> Figure:
    if subplots_kwargs is None:
        fig, (axis_ref, axis_query) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, (axis_ref, axis_query) = plt.subplots(
            1, 2, figsize=(12, 6), **subplots_kwargs
        )
    df = reference.merge(query, how="inner", on="name", suffixes=("_ref", "_query"))
    marker_sizes = df["cluster_size"] * s if "cluster_size" in df.columns else s
    for axis, postfix in ((axis_ref, "_ref"), (axis_query, "_query")):
        sns.scatterplot(
            data=df,
            x="x" + postfix,
            y="y" + postfix,
            hue="label" + postfix,
            s=marker_sizes,
            alpha=alpha,
            ax=axis,
            palette=palette,
            legend=False,
        )
        axis.set_xlabel("$X$")
        axis.set_ylabel("$Y$")
        axis.axis("equal")
    unique_labels = df["label_ref"].unique()
    num_unclustered = sum(df["label_ref"] == -1)
    for label in unique_labels:
        if label == -1:
            continue
        subset = df[df["label_ref"] == label]
        for axis, postfix in ((axis_ref, "_ref"), (axis_query, "_query")):
            x, y = bounding_square(subset["x" + postfix], subset["y" + postfix])
            axis.plot(x, y, "k:", linewidth=1.0)
            axis.text(max(x), max(y), label, fontsize=8, ha="left", va="bottom")
    fig.suptitle(
        f"{unique_labels.size} clusters, {num_unclustered} unclustered samples"
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Embed PHYLIP distance matrix using UMAP and cluster with DBSCAN."
    )
    parser.add_argument("reference", help="Reference tsv file")
    parser.add_argument("query", help="Query tsv file")
    parser.add_argument("-o", "--output", default="-", help="Output file")
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

    dpi = args.dpi
    try:
        dpi = float(dpi)
        if dpi <= 0.0:
            raise ValueError
    except ValueError:
        raise ValueError(f"invalid DPI value: {dpi!r}")
    fig = compare_embeddings(
        pd.read_csv(args.reference, sep="\t"), pd.read_csv(args.query, sep="\t")
    )
    fig.savefig(args.output, dpi=dpi)
    if args.interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
