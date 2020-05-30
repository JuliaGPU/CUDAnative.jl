using CSV
using DataFrames
using Plots

pyplot()

function plot_results(file, label)
    df = DataFrame(CSV.File(file))

    N = df[!, :N]
    mean_runtime = df[!, :runtime] .* 1e3 # in ps

    tflops = (8 .* N .^ 3) ./ mean_runtime

    plot!(N, tflops, label=label, xscale=:log2, markershape=:circle)
end

plot_results("cudanative.csv", "CUDAnative")
plot_results("cudanative-generic-fp32.csv", "CUDAnative generic (FP32)")
plot_results("cudanative-generic-fp16.csv", "CUDAnative generic (FP16)")
plot_results("cutlass.csv", "CUTLASS Example")

title!("Performance of mixed-precision complex GEMM\nProblem size: N x N x N")
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
