import os
import subprocess
import zipfile
import pandas as pd
import tqdm as tqdm


def main():
    for entry in os.scandir():
        if entry.name.endswith(".cluster.txt"):
            cluster = entry.name
            break
    else:
        raise FileNotFoundError("no *.cluster.txt file in current directory")
    cluster_name = os.path.splitext(cluster)[0]
    with open(cluster) as f:
        names = [line.strip() for line in f]
    with zipfile.ZipFile("sketches.zip") as zf:
        zf.extractall("sketches")
    manifest = pd.read_csv(
        "sketches/SOURMASH-MANIFEST.csv", comment="#", index_col="name"
    )
    args = ["sourmash", "sig", "merge", f"--output={cluster_name}.sig.gz"]
    with open(cluster) as f:
        locs = (manifest.loc[name, "internal_location"] for name in names)
        args.extend(f"sketches/{loc}" for loc in locs)
    subprocess.run(args, check=True)
