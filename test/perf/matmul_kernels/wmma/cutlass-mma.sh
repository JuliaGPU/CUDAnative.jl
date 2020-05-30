#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# < 1 ]]; then
    echo "Usage $0 <CUTLASS_BUILD_PATH>" 1>&2
    exit 1
fi

CUTLASS_BUILD_PATH=$1

cd "$( dirname "${BASH_SOURCE[0]}" )"

printf "N,runtime\n" >cutlass-mma.csv

for i in {7..14}; do
    N=$((2**i))

    # runtime in ns
    runtime=$(nv-nsight-cu-cli -f --summary per-kernel --csv --units base -k Kernel ${CUTLASS_BUILD_PATH}/tools/profiler/cutlass_profiler --m=$N --n=$N --k=$N --warmup-iterations=1 --profiling-iterations=10 --verification-enabled=false --kernels=cutlass_tensorop_s884gemm_f16_128x128_nn 2>/dev/null | grep 'gpu__time_duration' | tail -1 | awk -F',' '{print $NF}' | sed 's/"//g')

    printf "$N,$runtime\n" >>cutlass-mma.csv
done
