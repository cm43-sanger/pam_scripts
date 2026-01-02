import abc
import os
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_MIN_COUNT = 2
DEFAULT_NUM_THREADS = os.cpu_count() or 1
DEFAULT_MAX_MEMORY = 2.0


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("not positive integer")
    return value


def positive_float(value):
    value = float(value)
    if value < 1.0:
        raise ValueError("not positive float")
    return value


def call(args: Iterable[str]):
    pass


@dataclass
class BaseCounter(abc.ABC):
    min_count: positive_int = DEFAULT_MIN_COUNT
    num_threads: int = DEFAULT_NUM_THREADS
    max_memory: float = DEFAULT_MAX_MEMORY
    verbose: bool = True

    @abc.abstractmethod
    def count_kmers_args(
        self,
        db_name: str,
        min_count: int = DEFAULT_MIN_COUNT,
        num_threads: int = DEFAULT_NUM_THREADS,
        max_memory: float = DEFAULT_MAX_MEMORY,
        verbose: bool = True,
    ) -> Iterable[str]:
        pass

    def count_kmers(
        self,
        db_name: str,
        min_count: int = DEFAULT_MIN_COUNT,
        num_threads: int = DEFAULT_NUM_THREADS,
        max_memory: float = DEFAULT_MAX_MEMORY,
        verbose: bool = True,
    ):
        return call(
            self.count_kmers_args(
                db_name,
                min_count=min_count,
                num_threads=num_threads,
                max_memory=max_memory,
                verbose=verbose,
            )
        )
