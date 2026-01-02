import shutil
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: python cluster2sketch.py cluster", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as cluster, open(f"sequences.fa", "wb") as out_seq:
        for line in cluster:
            name, filename = line.strip().split(",")
            with open(filename, "rb") as in_seq:
                shutil.copyfileobj(in_seq, out_seq)
