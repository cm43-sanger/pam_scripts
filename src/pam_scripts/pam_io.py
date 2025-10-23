import numpy as np
import numpy.typing as npt
import sys
import typing
from collections.abc import Sequence
from contextlib import contextmanager


@contextmanager
def get_input_handle(filename: str = "-", mode: str = "r"):
    if filename == "-":
        f = sys.stdin
        if f.isatty():
            raise ValueError("Reading from stdio but no data piped")
        yield f
    else:
        with open(filename, mode) as f:
            yield f


@contextmanager
def get_output_handle(filename: str = "-", mode: str = "w"):
    if filename == "-":
        yield sys.stdout
    else:
        with open(filename, mode) as f:
            yield f


def load_distance_matrix(filename: str):
    """
    Load a PHYLIP distance matrix file (sequential/lower-triangular format).

    Parameters
    ----------
    file : str, os.PathLike, int, or file-like object
        Path to the PHYLIP file or an open file-like object.

    Returns
    -------
    names : list of str
        List of taxon names in file order.
    distances : numpy.ndarray
        Symmetric (num_taxa x num_taxa) matrix of pairwise distances.

    Raises
    ------
    ValueError
        If the file cannot be parsed, has incorrect dimensions, or contains
        invalid values. The error message includes the line number that
        caused the failure.

    PHYLIP sequential format:
        - First line: number of taxa (integer)
        - Each following line: taxon name, then distances to previously
            listed taxa
        - Only lower triangle is read; matrix is assumed symmetric
    """

    with get_input_handle(filename) as f:
        taxon_index = -1  # header line before first taxon
        try:
            num_taxa = int(next(f).strip())
            taxon_index += 1
            names = [next(f).strip().split(maxsplit=1)[0]]
            distances = np.zeros((num_taxa, num_taxa), dtype=np.float64)
            taxon_index += 1
            for line in f:
                data = line.strip().split(maxsplit=taxon_index + 1)
                names.append(data[0])
                row = np.array(data[1 : 1 + taxon_index], dtype=np.float64)
                distances[taxon_index, :taxon_index] = distances[
                    :taxon_index, taxon_index
                ] = row
                taxon_index += 1
            assert taxon_index == num_taxa
        except Exception as e:
            raise ValueError(
                f"Invalid phylip file: failed at line {taxon_index + 1}"
            ) from e
    return (names, distances)


import numpy as np


def write_distance_matrix(
    file: typing.TextIO, names: Sequence[str], distances: npt.ArrayLike
):
    """
    Write a PHYLIP distance matrix file (sequential/lower-triangular format).

    Parameters
    ----------
    filename : str or path-like
        Output file path.
    names : list of str
        Taxon names in order.
    distances : array-like
        Symmetric (num_taxa x num_taxa) distance matrix.

    Raises
    ------
    ValueError
        If names and matrix dimensions are inconsistent or matrix not symmetric.
    """

    num_taxa = len(names)
    distances = np.asarray(distances, dtype=float)
    if not (distances.ndim == 2 and distances.shape == (num_taxa, num_taxa)):
        raise ValueError(
            f"Distance matrix is not a square matrix of shape ({num_taxa}x{num_taxa})."
        )
    length = max(map(len, names))
    print(num_taxa, file=file)
    print(names[0], file=file)
    for i, name in enumerate(names[1:], start=1):
        formatted_distances = (format(d, ".6f") for d in distances[i, :i])
        print(name.ljust(length), *formatted_distances, sep=" ", file=file)
