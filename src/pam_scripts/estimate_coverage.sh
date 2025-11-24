#!/usr/bin/env bash

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <input_file> [cutoff]"
    echo "  <input_file> : histogram file containing kmer counts and frequencies"
    echo "  [cutoff]     : optional threshold between 0 and 1 (default 0.99)"
    exit 1
fi
file="$1"
if [ ! -f "$file" ]; then
    echo "Error: file '$file' does not exist or is not a regular file"
    exit 1
fi
cutoff="${2:-0.99}"
if awk -v cutoff="$cutoff" '
    BEGIN {
        exit (cutoff > 0 && cutoff < 1)
    }
    '; then
    echo "Error: cutoff must be a number between 0 and 1"
    exit 1
fi

read total target <<< $(awk -v cutoff="$cutoff" '
{
    total += $1 * $2
}
END {
    total = total + 0
    target = total * cutoff
    print total, target
}
' "$file")

read high_count high_total high_unique low_count <<< $(awk -v target="$target" '
BEGIN { 
    high_count = 0 
}
{
    high_total += $1 * $2
    high_unique += $2
    if (high_total >= target) {
        high_count = $1
        exit
    }
}
END {
    high_count = high_count + 0
    ix = int(sqrt(high_count))
    low_count = ix + (ix * ix < high_count)
    print high_count, high_total + 0, high_unique + 0, low_count
}
' "$file")

read low_total low_unique <<< $(awk -v low_count="$low_count" '
{
    if ($1 >= low_count) {
        exit
    }
    low_total += $1 * $2
    low_unique += $2
}
END {
    print low_total + 0, low_unique + 0
    }
' "$file")

if [ "$low_unique" -eq "$high_unique" ]; then
    echo "NaN"
else
    active_unique=$((high_unique - low_unique))
    active_total=$((high_total - low_total))
    awk -v active_total="$active_total" -v active_unique="$active_unique" '
    BEGIN {
        printf "%.6f\n", active_total / active_unique
    }
    '
fi
