import argparse
import multiprocessing
import os
import sys
import traceback
import typing
import h5py
import numpy as np
from tempfile import TemporaryDirectory
from tqdm import tqdm as make_progressbar
from . import kmc, xxhash

DEFAULT_THRESHOLD = 0.05
DEFAULT_SEED = 42
UINT64_MAX = 2**64 - 1


class SketchHelper(kmc.KMCHelper):
    _scale: typing.Optional[int]
    _method: str = "custom2"
    _seed: int

    def __init__(
        self,
        kmer_length: int = kmc.DEFAULT_KMER_LENGTH,
        scale: typing.Optional[int] = None,
        seed: int = DEFAULT_SEED,
        max_memory: typing.Optional[float] = None,
        num_threads: typing.Optional[int] = None,
    ):
        self.kmer_length = kmer_length
        self.correct_errors = True
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
                if value <= 0:
                    raise ValueError
            except ValueError:
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
            if value <= 0 or value > UINT64_MAX:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"seed must be an integer in range [1, 2^64-1] (got {value})"
            )
        self._seed = value

    def save_config(self, file: h5py.File):
        info = file.create_group("info")
        info.create_dataset("kmer_length", data=np.uint8(self.kmer_length))
        info.create_dataset("scale", data=np.uint64(self.scale))
        info.create_dataset("method", data=self.method)
        info.create_dataset("seed", data=np.uint64(self.seed))
        return info

    def sketch_reads(self, *reads: str) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
        with TemporaryDirectory() as temporary_directory:
            db_path = os.path.join(temporary_directory, "counts")
            db = self.count_kmers(db_path, *reads)
            kmers = db.load_kmers()
        hashes = xxhash.hash_kmers(kmers, seed=self.seed, num_threads=self.num_threads)
        if self.scale != 1:
            max_value = UINT64_MAX // self.scale
            passed = hashes <= max_value
            hashes = hashes[passed]
        hashes.sort()
        return hashes


def load_manifest(manifest: str):
    unique_names: set[str] = set()
    samples: list[tuple[str, list[str]]] = []
    try:
        with open(manifest) as f:
            for i, line in enumerate(f, start=1):
                try:
                    name, *reads = map(str.strip, line.strip().split("\t"))
                    if not reads:
                        raise ValueError("no reads specified")
                except Exception as e:
                    raise ValueError(f"line {i} is invalid: {line.strip()!r}") from e
                if name in unique_names:
                    raise ValueError(f"repeated name {name!r} in line {i}")
                unique_names.add(name)
                for read in reads:
                    if not os.path.exists(read):
                        raise FileNotFoundError(f"{read!r} in line {i}")
                samples.append((name, reads))
    except Exception as e:
        raise ValueError(f"unable to load manifest {manifest!r}") from e
    return samples


class SketchResult(typing.NamedTuple):
    name: str
    reads: list[str]
    success: int
    hashes: typing.Optional[np.ndarray[tuple[int], np.dtype[np.uint64]]] = None
    message: str = ""


__sketch_from_manifest_helper: typing.Optional[SketchHelper] = None


def __sketch_from_manifest_worker_init(sketch_helper: SketchHelper):
    global __sketch_from_manifest_helper
    __sketch_from_manifest_helper = sketch_helper


def __sketch_from_manifest_worker_func(samples: tuple[str, list[str]]):
    if __sketch_from_manifest_helper is None:
        raise RuntimeError(
            "worker function called outside of initialized multiprocessing context."
        )
    name, reads = samples
    try:
        hashes = __sketch_from_manifest_helper.sketch_reads(*reads)
    except Exception as e:
        error_message = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return SketchResult(name, reads, success=False, message=error_message)
    return SketchResult(name, reads, success=True, hashes=hashes)


class ResolvedArguments(typing.NamedTuple):
    num_threads: int
    num_jobs: int
    num_job_threads: int
    compression_level: int


def resolve_numerical_arguments(
    num_threads: typing.Optional[int] = None,
    num_jobs: typing.Optional[int] = None,
    compression_level: int = 4,
):
    try:
        num_threads = kmc.NUM_CPUS if num_threads is None else int(num_threads)
        if num_threads <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"num_threads must be a positive integer (got {num_threads})")
    try:
        num_jobs = 1 if num_jobs is None else int(num_jobs)
        if num_jobs <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"num_jobs must be a positive integer (got {num_jobs})")
    try:
        compression_level = int(compression_level)
        if compression_level <= 0 or compression_level >= 10:
            raise ValueError
    except ValueError:
        raise ValueError(
            "compression_level must be an integer in range [1, 9] "
            f"(got {compression_level})"
        )
    num_jobs = min(num_jobs, num_threads)
    num_job_threads = (num_threads - 1) // num_jobs + 1
    return ResolvedArguments(
        num_threads=num_threads,
        num_jobs=num_jobs,
        num_job_threads=num_job_threads,
        compression_level=compression_level,
    )


def resolve_output_file(output_filename: str, overwrite: bool = False):
    if not output_filename.lower().endswith(".h5"):
        output_filename = output_filename + ".h5"
    output_filename = os.path.abspath(output_filename)
    if os.path.exists(output_filename):
        if os.path.isdir(output_filename):
            raise IsADirectoryError(output_filename)
        if not overwrite:
            raise FileExistsError(output_filename)
    else:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    return output_filename


def sketch_from_manifest(
    manifest: str,
    output_filename: str,
    kmer_length: int = kmc.DEFAULT_KMER_LENGTH,
    scale: typing.Optional[int] = None,
    seed: int = DEFAULT_SEED,
    max_memory: typing.Optional[float] = None,
    num_threads: typing.Optional[int] = None,
    num_jobs: typing.Optional[int] = None,
    compression_level: int = 4,
    overwrite: bool = False,
    verbose: bool = False,
):
    args = resolve_numerical_arguments(
        num_threads=num_threads, num_jobs=num_jobs, compression_level=compression_level
    )
    output_filename = resolve_output_file(output_filename, overwrite=overwrite)
    helper = SketchHelper(
        kmer_length=kmer_length,
        scale=scale,
        seed=seed,
        max_memory=max_memory,
        num_threads=args.num_job_threads,
    )
    samples = load_manifest(manifest)
    if verbose:
        print(
            f"Sketching {len(samples)} samples from {manifest!r} "
            f"to {output_filename!r} with {args.num_jobs} jobs, "
            f"each with {helper.num_threads} threads.",
            file=sys.stderr,
        )
    failures: list[str] = []
    with (
        multiprocessing.Pool(
            num_jobs, initializer=__sketch_from_manifest_worker_init, initargs=(helper,)
        ) as pool,
        make_progressbar(
            pool.imap_unordered(__sketch_from_manifest_worker_func, samples),
            desc="Sketching",
            total=len(samples),
            disable=not verbose,
            postfix={"failures": 0},
        ) as progressbar,
        h5py.File(output_filename, "w") as f,
    ):
        helper.save_config(f)
        data = f.create_group("data")
        for result in progressbar:
            if result.success:
                assert result.hashes is not None  # otherwise pylance complains
                data.create_dataset(
                    result.name,
                    data=result.hashes,
                    compression="gzip",
                    compression_opts=args.compression_level,
                    shuffle=True,  # transpose bytes for better compression
                )
            else:
                failures.append(result.name)
                progressbar.set_postfix({"failures": len(failures)})
                if verbose:
                    error_message = (
                        f"{result.message}Error processing {result.name!r} "
                        f"({result.reads})\n"
                    )
                    progressbar.write(error_message)
    if verbose and failures:
        print(f"Failed to sketch the following samples: {failures}", file=sys.stderr)
    return len(samples) - len(failures)


def load_sketches(path: str, scale: typing.Optional[int] = None):
    names: list[str] = []
    sketches: list[np.ndarray[tuple[int], np.dtype[np.uint64]]] = []
    with h5py.File(path, "r") as f:
        sketch_scale = int(f["info"]["scale"][()])
        try:
            scale = sketch_scale if scale is None else int(scale)
            if scale < sketch_scale:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Provided scale ({scale}) not an integer >= sketch scale "
                f"({sketch_scale})"
            )
        if scale == sketch_scale:
            for name, data in f["data"].items():
                print(name)
                names.append(name)
                sketches.append(np.asarray(data[:], dtype=np.uint64))
        else:
            cutoff = UINT64_MAX // scale
            for name, data in f["data"].items():
                print(name)
                names.append(name)
                raw_hashes = np.asarray(data[:], dtype=np.uint64)
                cutoff = np.searchsorted(raw_hashes, cutoff)
                sketches.append(raw_hashes[:cutoff].copy())
    return (names, sketches)


def main():
    parser = argparse.ArgumentParser(
        description="Generate kmer sketches from a manifest of read sets"
    )
    parser.add_argument(
        "manifest",
        help="Path to the manifest file with columns for name and each read "
        "(tab-separated, no header, names must be unique)",
    )
    parser.add_argument(
        "output_filename",
        help="Output filename for the generated sketches "
        "(.h5 extension is appended if not present)",
    )
    parser.add_argument(
        "-k",
        "--kmer_length",
        type=int,
        default=kmc.DEFAULT_KMER_LENGTH,
        help=f"Kmer length (default {kmc.DEFAULT_KMER_LENGTH}, odd, "
        f">={kmc.MINIMUM_KMER_LENGTH}, <={kmc.MAXIMUM_KMER_LENGTH})",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=None,
        help="Downsampling scale factor (default: kmer_length)",
    )
    parser.add_argument(
        "-d",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic seed for hash function (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "-m",
        "--max_memory",
        type=float,
        default=kmc.MINIMUM_MAX_MEMORY,
        help=f"Max amount of RAM in GB (default {kmc.MINIMUM_MAX_MEMORY}, "
        f">={kmc.MINIMUM_MAX_MEMORY})",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=kmc.NUM_CPUS,
        help=f"Total number of threads (default {kmc.NUM_CPUS})",
    )
    parser.add_argument(
        "-j",
        "--num-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    parser.add_argument(
        "-z",
        "--compression_level",
        type=int,
        default=4,
        choices=range(1, 10),
        metavar="[1-9]",
        help="Gzip compression level for HDF5 sketches (1-9, default 4)",
    )
    parser.add_argument(
        "-f", "--overwrite", action="store_true", help="Overwrite existing file"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose progress output"
    )
    args = parser.parse_args()

    sketch_from_manifest(
        args.manifest,
        args.output_filename,
        kmer_length=args.kmer_length,
        scale=args.scale,
        seed=args.seed,
        max_memory=args.max_memory,
        num_threads=args.num_threads,
        num_jobs=args.num_jobs,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
