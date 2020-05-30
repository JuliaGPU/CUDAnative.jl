#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# < 1 ]]; then
    echo "Usage $0 <JULIA_PATH>" 1>&2
    exit 1
fi

JULIA_PATH=$1

cd "$( dirname "${BASH_SOURCE[0]}" )"

printf "N,runtime\n" >cudanative-generic-fp16.csv

for i in {7..14}; do
    N=$((2**i))

    # runtime in ns
    runtime=$(LD_LIBRARY_PATH=${JULIA_PATH}/usr/lib nv-nsight-cu-cli --profile-from-start off -f --summary per-kernel --csv --units base ${JULIA_PATH}/julia cudanative-generic.jl $N $N $N FP16 2>/dev/null | grep 'gpu__time_duration' | tail -1 | awk -F',' '{print $NF}' | sed 's/"//g')

    printf "$N,$runtime\n" >>cudanative-generic-fp16.csv
done
