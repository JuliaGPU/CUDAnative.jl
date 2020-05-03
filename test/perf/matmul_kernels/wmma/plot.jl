using CSV
using DataFrames
using Plots

pyplot()

function plot_results(file, label)
    df = DataFrame(CSV.File(file))

    N = df[!, :N]
    mean_runtime = df[!, :runtime] .* 1e3 # in ps

    tflops = (2 .* N .^ 3) ./ mean_runtime

    plot!(N, tflops, label=label, xscale=:log2, markershape=:circle)
end

plot_results("cudanative.csv", "CUDAnative")
plot_results("cudanative-generic-fp32.csv", "CUDAnative generic (FP32)")
plot_results("cudanative-generic-fp16.csv", "CUDAnative generic (FP16)")
plot_results("cublas.csv", "cuBLAS")
plot_results("cutlass-wmma.csv", "CUTLASS (WMMA)")
plot_results("cutlass-mma.csv", "CUTLASS (mma.m8n8k4)")
plot_results("cutlass-mma-turing.csv", "CUTLASS (mma.m16n8k8)")

title!("Performance of mixed-precision GEMM\nProblem size: N x N x N")
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
