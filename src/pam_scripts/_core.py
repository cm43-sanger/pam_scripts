import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass


NUM_CPUS = os.cpu_count() or 1


class DirectConstructionError(UserWarning):
    pass


warnings.simplefilter("error", DirectConstructionError)


def __post_init__(self):
    warnings.warn(
        f"don't create {self.__class__.__name__} directly", DirectConstructionError
    )


def guarded_dataclass(cls):
    cls.__post_init__ = __post_init__
    return dataclass(frozen=True)(cls)


@contextmanager
def release_guard():
    with warnings.catch_warnings(action="ignore", category=DirectConstructionError):
        yield
