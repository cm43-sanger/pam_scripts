import shutil

for executable in ("kmc", "kmc_tools"):
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Required executable '{executable}' not found in PATH."
        )

import os
import subprocess
import typing
from collections.abc import Iterable

MAX_THREADS = 128
NUM_CPUS = os.cpu_count() or 1


def _resolve_num_threads(num_threads: typing.Optional[int]):
    if num_threads is None:
        num_threads = NUM_CPUS
    if num_threads < 1:
        raise ValueError("number of threads must be positive")
    return min(num_threads, MAX_THREADS)


def _call_executable(
    executable: str, args: Iterable[str], num_threads: typing.Optional[int] = None
):
    num_threads = _resolve_num_threads(num_threads)
    subprocess_args = [executable, f"-t{num_threads}"]
    subprocess_args.extend(args)
    print(subprocess_args)
    # result = subprocess.run(subprocess_args, capture_output=True)
    result = subprocess.run(subprocess_args)
    if result.returncode:
        raise RuntimeError(
            f"\n{subprocess_args}"
            f"\nfailed with exit code {result.returncode}. stderr:"
            f"\n{result.stderr.decode().strip()}"
        )
    return result


def call_kmc(args: Iterable[str], num_threads: typing.Optional[int] = None):
    return _call_executable("kmc", args, num_threads=num_threads)


def call_kmc_tools(args: Iterable[str], num_threads: typing.Optional[int] = None):
    return _call_executable("kmc_tools", args, num_threads=num_threads)
