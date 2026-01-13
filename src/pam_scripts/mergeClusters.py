import os
import subprocess
import zipfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def process_cluster(manifest, label, rows):
    with open(f"clusters/{label}.csv", "w") as f:
        for name, filename in manifest.loc[rows["name"], "filename"].items():
            print(name, filename, sep=",", file=f)
    paths = [
        f"manysketch/{loc}" for loc in manifest.loc[rows["name"], "internal_location"]
    ]
    return (label, paths)


def merge_signatures(cluster):
    label, paths = cluster
    filename = f"signatures/{label}.sig.gz"
    args = [
        "sourmash",
        "sig",
        "merge",
        "--quiet",
        f"--output={filename}",
        f"--set-name={label}",
    ]
    args.extend(paths)
    subprocess.run(args, check=True)
    return filename


def index_clusters(clusters):
    args = ["sourmash", "index", "clusters.sbt.zip"]
    with ThreadPoolExecutor() as executor:
        args.extend(executor.map(merge_signatures, clusters))
    subprocess.run(args, check=True)


def main():
    os.makedirs("clusters")
    os.makedirs("signatures")
    with zipfile.ZipFile("manysketch.zip") as zf:
        zf.extractall("manysketch")
    manifest = pd.read_csv(
        "manysketch/SOURMASH-MANIFEST.csv", comment="#", index_col="name"
    )
    embedding = pd.read_csv("embedding.tsv", sep="\t")
    embedding_clusters = embedding.drop_duplicates(
        subset=("label", "sub_label")
    ).groupby("label")
    clusters = (
        process_cluster(manifest, label, rows)
        for label, rows in embedding_clusters
        if label != "unclustered"
    )
    index_clusters(clusters)
