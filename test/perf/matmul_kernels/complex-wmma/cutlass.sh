#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# < 1 ]]; then
    echo "Usage $0 <CUTLASS_BUILD_PATH>" 1>&2
    exit 1
fi

CUTLASS_BUILD_PATH=$1

cd "$( dirname "${BASH_SOURCE[0]}" )"

printf "N,runtime\n" >cutlass.csv

for i in {7..14}; do
    N=$((2**i))

    # runtime in ns
    runtime=$(nv-nsight-cu-cli -f --summary per-kernel --csv --units base -k Kernel ${CUTLASS_BUILD_PATH}/examples/10_planar_complex/10_planar_complex --m=$N --n=$N --k=$N 2>/dev/null | grep 'gpu__time_duration' | tail -1 | awk -F',' '{print $NF}' | sed 's/"//g')

    printf "$N,$runtime\n" >>cutlass.csv
done
