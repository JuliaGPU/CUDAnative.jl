#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# < 1 ]]; then
    echo "Usage $0 <CUTLASS_BUILD_PATH>" 1>&2
    exit 1
fi

CUTLASS_BUILD_PATH=$1

cd "$( dirname "${BASH_SOURCE[0]}" )"

printf "N,runtime\n" >cutlass-mma-turing.csv

for i in {7..14}; do
    N=$((2**i))

    # runtime in ns
    runtime=$(nv-nsight-cu-cli -f --summary per-kernel --csv --units base -k Kernel ${CUTLASS_BUILD_PATH}/tools/profiler/cutlass_profiler --op_class=tensorop --A=f16:col --B=f16:col --C=f32 --accum=f32 --m=$N --n=$N --k=$N --inst_m=16 --inst_n=8 --inst_k=8 --warmup-iterations=1 --profiling-iterations=10 2>/dev/null | grep 'gpu__time_duration' | tail -1 | awk -F',' '{print $NF}' | sed 's/"//g')

    printf "$N,$runtime\n" >>cutlass-mma-turing.csv
done
