from . import kmc, kmers, xxhash

import argparse
import h5py
import json
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil
import sys
import threading
import time
import typing
import warnings
from collections.abc import Sequence
from numba import njit, prange
from tempfile import TemporaryDirectory
from tqdm import tqdm as make_progressbar

DEFAULT_THRESHOLD = 0.05
DEFAULT_SEED = 42
UINT64_MAX = 2**64 - 1


class SketchHelper(kmc.KMCHelper):
    _scale: typing.Optional[int]
    _method: str = "custom"
    _seed: int

    def __init__(
        self,
        kmer_length: int = kmc.DEFAULT_KMER_LENGTH,
        threshold: float = DEFAULT_THRESHOLD,
        scale: typing.Optional[int] = None,
        seed: int = DEFAULT_SEED,
        max_memory: typing.Optional[float] = None,
        num_threads: typing.Optional[int] = None,
    ):
        self.kmer_length = kmer_length
        self.threshold = threshold
        self.scale = scale
        self.seed = seed
        self.max_memory = max_memory
        self.num_threads = num_threads

    @property
    def scale(self):
        return self.kmer_length if self._scale is None else self._scale

    @scale.setter
    def scale(self, value: typing.Optional[int] = None):
        if value is not None:
            try:
                value = int(value)
                assert value > 0
            except:
                raise ValueError(f"scale must be a positive integer (got {value})")
        self._scale = value

    @property
    def method(self):
        return self._method

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        try:
            value = int(value)
            assert value > 0 and value <= UINT64_MAX
        except:
            raise ValueError(
                f"seed must be an integer in range [1, 2^64-1] (got {value})"
            )
        self._seed = value

    def save_config(self, path: str):
        config = {
            "kmer_length": self.kmer_length,
            "threshold": round(self.threshold, 6),
            "scale": self.scale,
            "method": self.method,
            "seed": self.seed,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def sketch_reads(
        self, read1: str, read2: str
    ) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
        with TemporaryDirectory() as temporary_directory:
            db_path = os.path.join(temporary_directory, "counts")
            db = self.count_reads(read1, read2, db_path)
            kmers = db.load_kmers()
        if self.scale != 1:
            hashes = xxhash.hash_kmers(
                kmers, seed=self.seed, num_threads=self.num_threads
            )
            max_count = UINT64_MAX // self.scale
            passed = hashes <= max_count
            kmers = kmers[passed]
        kmers.sort()
        return kmers


def load_manifest(manifest: str):
    unique_names: set[str] = set()
    samples: list[tuple[str, str, str]] = []
    try:
        with open(manifest) as f:
            for i, line in enumerate(f, start=1):
                try:
                    name, read1, read2 = map(str.strip, line.strip().split("\t"))
                except ValueError:
                    raise ValueError(f"line {i} is invalid: {line.strip()!r}")
                if name in unique_names:
                    raise ValueError(f"repeated name {name!r} in line {i}")
                unique_names.add(name)
                if not os.path.exists(read1):
                    raise FileNotFoundError(f"{read1!r} in line {i}")
                if not os.path.exists(read2):
                    raise FileNotFoundError(f"{read2!r} in line {i}")
                samples.append((name, read1, read2))
    except Exception as e:
        raise ValueError(f"unable to load manifest {manifest!r}") from e
    return samples


__sketch_from_manifest_helper: typing.Optional[SketchHelper] = None


def __sketch_from_manifest_worker_init(sketch_helper: SketchHelper):
    global __sketch_from_manifest_helper
    __sketch_from_manifest_helper = sketch_helper


def __sketch_from_manifest_worker_func(samples: tuple[str, str, str]):
    if __sketch_from_manifest_helper is None:
        raise RuntimeError(
            "worker function called outside of initialized multiprocessing context."
        )
    name, read1, read2 = samples
    try:
        sketch = __sketch_from_manifest_helper.sketch_reads(read1, read2)
    except:
        return (name, read1, read2, None)
    return (name, read1, read2, sketch)


class ResolvedThreads(typing.NamedTuple):
    num_threads: int
    num_jobs: int
    num_job_threads: int


def resolve_threads(
    num_threads: typing.Optional[int] = None,
    num_jobs: typing.Optional[int] = None,
):
    try:
        num_threads = kmc.NUM_CPUS if num_threads is None else int(num_threads)
        assert num_threads > 0
    except:
        raise ValueError(f"num_threads must be a positive integer (got {num_threads})")
    try:
        num_jobs = 1 if num_jobs is None else int(num_jobs)
        assert num_jobs > 0
    except:
        raise ValueError(f"num_jobs must be a positive integer (got {num_jobs})")
    num_jobs = min(num_jobs, num_threads)
    num_job_threads = (num_threads - 1) // num_jobs + 1
    return ResolvedThreads(
        num_threads=num_threads, num_jobs=num_jobs, num_job_threads=num_job_threads
    )


def sketch_from_manifest(
    manifest: str,
    output_directory: str,
    kmer_length: int = kmc.DEFAULT_KMER_LENGTH,
    threshold: float = DEFAULT_THRESHOLD,
    scale: typing.Optional[int] = None,
    seed: int = DEFAULT_SEED,
    max_memory: typing.Optional[float] = None,
    num_threads: typing.Optional[int] = None,
    num_jobs: typing.Optional[int] = None,
    exist_ok: bool = False,
    verbose: bool = False,
):
    resolved_threads = resolve_threads(num_threads=num_threads, num_jobs=num_jobs)
    helper = SketchHelper(
        kmer_length=kmer_length,
        threshold=threshold,
        scale=scale,
        seed=seed,
        max_memory=max_memory,
        num_threads=resolved_threads.num_job_threads,
    )
    samples = load_manifest(manifest)
    if verbose:
        print(
            f"Sketching {len(samples)} paired-end reads from {manifest!r} "
            f"to {output_directory!r} with {resolved_threads.num_jobs} jobs, "
            f"each with {helper.num_threads} threads.",
            file=sys.stderr,
        )
    if os.path.exists(output_directory):
        if not exist_ok:
            raise FileExistsError(
                f"output directory {output_directory!r} already exists"
            )
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    helper.save_config(os.path.join(output_directory, "config.json"))
    num_failures = 0
    with (
        multiprocessing.Pool(
            num_jobs,
            initializer=__sketch_from_manifest_worker_init,
            initargs=(helper,),
        ) as pool,
        make_progressbar(
            pool.imap_unordered(__sketch_from_manifest_worker_func, samples),
            desc="Sketching",
            total=len(samples),
            disable=not verbose,
            postfix={"failures": 0},
        ) as progressbar,
        open(os.path.join(output_directory, "results.tsv"), "w") as tsv_fp,
        h5py.File(os.path.join(output_directory, "sketches.h5"), "w") as h5_fp,
    ):
        print("name", "read1", "read2", "success", sep="\t", file=tsv_fp)
        for name, read1, read2, kmers in progressbar:
            success = kmers is not None
            num_failures += not success
            print(name, read1, read2, success, sep="\t", file=tsv_fp)
            if success:
                h5_fp.create_dataset(name, data=kmers, compression="gzip")
            progressbar.set_postfix({"failures": num_failures})
            progressbar.update()
    return len(samples) - num_failures


# def __load_sketches_worker_func(filename: str):
#     return kmers.load_kmers(filename, num_threads=1)


# @njit
# def _jaccard_similarity_numba(a, b):
#     """Compute Jaccard similarity between two sorted uint64 arrays."""
#     i = 0
#     j = 0
#     intersection = 0
#     len_a = a.size
#     len_b = b.size
#     while i < len_a and j < len_b:
#         ai = a[i]
#         bj = b[j]
#         intersection += ai == bj
#         i += ai <= bj
#         j += ai >= bj
#     union = len_a + len_b - intersection
#     if union == 0:
#         return 0.0
#     return intersection / union


# @njit(parallel=True)
# def _pairwise_jaccard_numba(n, arrays, d, progress):
#     # Compute upper triangle in parallel
#     for i in prange(n):
#         d[i, i] = 1.0  # diagonal
#         for j in range(i + 1, n):
#             sim = _jaccard_similarity_numba(arrays[i], arrays[j])
#             d[i, j] = sim
#             d[j, i] = sim  # symmetric
#         progress[0] += i


# def pairwise_jaccard(arrays):
#     """
#     Compute pairwise Jaccard similarity between a list of sorted uint64 arrays.
#     Returns a symmetric float64 matrix.
#     """
#     n = len(arrays)
#     d = np.empty((n, n), dtype=np.float64)
#     total = n * (n - 1) // 2
#     with make_progressbar(total=total) as progressbar:
#         progress = np.zeros(1, dtype=np.int64)
#         thread = threading.Thread(
#             target=_pairwise_jaccard_numba, args=(n, arrays, d, progress)
#         )
#         thread.start()
#         last = 0
#         while thread.is_alive():
#             current = progress[0]
#             progressbar.update(current - last)
#             last = current
#             time.sleep(0.01)
#         thread.join()
#     return d


# def load_sketches(directory: str):
#     results = pd.read_csv(os.path.join(directory, "results.tsv"), sep="\t")
#     unsuccessful_names = []
#     names = []
#     sketches_directory = os.path.join(directory, "sketches")
#     for row in results.itertuples():
#         if row.success:
#             names.append(row.name)
#         else:
#             unsuccessful_names.append(row.name)
#     filenames = (os.path.join(sketches_directory, name) for name in names)
#     with (
#         multiprocessing.Pool() as pool,
#         make_progressbar(
#             pool.imap(__load_sketches_worker_func, filenames),
#             desc="Loading databases",
#             total=len(names),
#         ) as progressbar,
#     ):
#         kmers_list_lookup = {
#             name: kmers_list for name, kmers_list in zip(names, progressbar)
#         }
#     results["kmers"] = results["name"].map(kmers_list_lookup)
#     return results


# def calculate_distances(results: pd.DataFrame):
#     mask =
#     num = results['']


def main():
    parser = argparse.ArgumentParser(
        description="Generate kmer sketches from a manifest of read sets."
    )
    parser.add_argument(
        "manifest",
        help="Path to the manifest file listing input read sets (TSV format)",
    )
    parser.add_argument(
        "directory", help="Output directory to store the generated sketches"
    )
    parser.add_argument(
        "--kmer-length",
        "-k",
        type=int,
        default=21,
        help="K-mer length to use for sketching (default: 21)",
    )
    parser.add_argument(
        "--num-jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel jobs (default: auto)",
    )
    parser.add_argument(
        "--num-kmc-threads",
        "-t",
        type=int,
        default=None,
        help="Threads per job for KMC (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose progress output"
    )
    args = parser.parse_args()

    sketch_from_manifest(
        args.manifest,
        args.directory,
        kmer_length=args.kmer_length,
        num_jobs=args.num_jobs,
        num_kmc_threads=args.num_kmc_threads,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
