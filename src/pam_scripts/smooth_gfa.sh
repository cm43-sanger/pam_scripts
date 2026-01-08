#!/usr/bin/env bash
set -euo pipefail

graph="$1"
num="$2"

read r < "$num"

smoothxg -t11 -r"$r" "$graph" -o smooth_cat2.gfa
