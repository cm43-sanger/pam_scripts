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


def plot_embedding(
    embedding: pd.DataFrame,
    name: typing.Optional[str] = None,
    s: float = 8.0,
    alpha: float = 0.8,
    palette: str = "tab10",
    show_counts: bool = False,
    subplots_kwargs: typing.Optional[dict] = None,
) -> Figure:
    if subplots_kwargs is None:
        fig, axis = plt.subplots()
    else:
        fig, axis = plt.subplots(**subplots_kwargs)
    marker_sizes = (
        embedding["cluster_size"] * s if "cluster_size" in embedding.columns else s
    )
    sns.scatterplot(
        data=embedding,
        x="x",
        y="y",
        hue="label",
        s=marker_sizes,
        alpha=alpha,
        ax=axis,
        palette=palette,
        legend=False,
    )
    unique_labels = embedding["label"].unique()
    num_unclustered = sum(embedding["label"] == -1)
    for label in unique_labels:
        if label == -1:
            continue
        subset = embedding[embedding["label"] == label]
        num_unique_sub_labels = subset["sub_label"].nunique()
        is_singleton = num_unique_sub_labels == 1
        x, y = bounding_square(subset["x"], subset["y"])
        axis.plot(x, y, "k:" if is_singleton else "k-", linewidth=1.0)
        if show_counts:
            axis.text(
                max(x),
                max(y),
                str(len(subset)) if show_counts else label,
                fontsize=8,
                ha="left",
                va="bottom",
            )
    axis.set_xlabel("$X$")
    axis.set_ylabel("$Y$")
    axis.axis("equal")
    phrase = f"{name} " if name else ""
    fig.suptitle(
        f"{len(embedding)} {phrase}samples in {unique_labels.size} clusters"
        # f"\n({num_unclustered} unclustered samples)"
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Embed PHYLIP distance matrix using UMAP and cluster with DBSCAN."
    )
    parser.add_argument(
        "input_tsv", nargs="?", default="-", help="Input tsv file (defaults to stdin)"
    )
    parser.add_argument("-o", "--output", default="-", help="Output file")
    parser.add_argument(
        "-d",
        "--dpi",
        default=DEFAULT_DPI,
        type=float,
        help=f"DPI for output PNG (default {DEFAULT_DPI})",
    )
    parser.add_argument(
        "-e",
        "--interactive",
        action="store_true",
        help="Show interactive plot of the embedding",
    )
    parser.add_argument(
        "-c", "--counts", action="store_true", help="Show cluster counts instead of ID"
    )
    parser.add_argument("-t", "--name", default=None, type=str, help="Species name")
    args = parser.parse_args()

    dpi = args.dpi
    try:
        dpi = float(dpi)
        if dpi <= 0.0:
            raise ValueError
    except ValueError:
        raise ValueError(f"invalid DPI value: {dpi!r}")
    with pam_io.get_input_handle(args.input_tsv) as f:
        embedding = pd.read_csv(f, sep="\t")
    fig = plot_embedding(embedding, name=args.name, show_counts=args.counts)
    fig.savefig(args.output, dpi=dpi)
    if args.interactive:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
