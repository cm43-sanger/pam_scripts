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
    args = [
        "sourmash",
        "sig",
        "merge",
        "--quiet",
        f"--output=signatures/{label}.sig.gz",
        f"--set-name={label}",
    ]
    args.extend(paths)
    subprocess.run(args, check=True)
    return label


def main():
    os.makedirs("clusters")
    os.makedirs("signatures")
    with zipfile.ZipFile("manysketch.zip") as zf:
        zf.extractall("manysketch")
    manifest = pd.read_csv(
        "manysketch/SOURMASH-MANIFEST.csv", comment="#", index_col="name"
    )
    embedding = pd.read_csv("embedding.tsv", sep="\t").drop_duplicates("label")
    embedding["label"] = embedding["label"].astype(str).str.split(".", n=1).str[0]
    embedding_clusters = embedding.groupby("density_label")
    for label, rows in embedding_clusters:
        process_cluster(manifest, label, rows)
    clusters = (
        process_cluster(manifest, label, rows)
        for label, rows in embedding_clusters
        if label != "unclustered"
    )
    with ThreadPoolExecutor() as executor:
        for label in executor.map(merge_signatures, clusters):
            pass
