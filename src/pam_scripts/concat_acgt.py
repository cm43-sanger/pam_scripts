import operator
import os
import re
import shutil
import sys
import typing
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from tempfile import TemporaryDirectory
from tqdm import tqdm as make_progressbar

AMBIGUOUS_REGEX = "[^ACGT]"  # any base other than ACGT

__working_directory: typing.Optional[str] = None


def worker_init(working_directory: str):
    global __working_directory
    __working_directory = working_directory


def worker_func(row: tuple[str, str]):
    if __working_directory is None:
        raise RuntimeError(
            'initialize working directory with "worker_init" before running "worker_func"'
        )
    name, path = row
    fa_path = os.path.join(__working_directory, f"{name}.fa")
    with open(fa_path, "w") as fa:
        for contig, record in enumerate(SeqIO.parse(path, "fasta"), start=1):
            frag = 1
            for seq in re.split(AMBIGUOUS_REGEX, str(record.seq).upper()):
                if len(seq) < 300:
                    continue
                SeqIO.write(
                    SeqRecord(Seq(seq), id=f"{name}#{contig}_{frag}", description=""),
                    fa,
                    "fasta",
                )
                frag += 1
    return fa_path


def main():
    if len(sys.argv) != 3:
        print(f"Usage: concat_acgt manifest.csv out.fa", file=sys.stderr)
        return 1
    manifest_path = sys.argv[1]
    out_path = sys.argv[2]
    input_files: dict[str, str] = {}
    with open(manifest_path) as manifest:
        for line in manifest:
            line = line.strip()
            if not line:
                continue
            name, path = line.split(",")
            if name in input_files:
                raise ValueError(f"duplicate name: {name!r}")
            input_files[name] = path
    ordered_rows = sorted(input_files.items(), key=operator.itemgetter(0))
    with (
        TemporaryDirectory() as working_directory,
        Pool(initializer=worker_init, initargs=(working_directory,)) as pool,
        make_progressbar(
            pool.imap(worker_func, ordered_rows), "Processing files"
        ) as progressbar,
        open(out_path, "wb") as out,
    ):
        for fa_path in progressbar:
            with open(fa_path, "rb") as fa:
                shutil.copyfileobj(fa, out)
            os.remove(fa_path)
    return 0
