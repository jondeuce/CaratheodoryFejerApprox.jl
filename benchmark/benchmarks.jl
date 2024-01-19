using Pkg: Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)

using BenchmarkTools
using CaratheodoryFejerApprox
using CaratheodoryFejerApprox.ApproxFun

function make_bench(f::F, dom::NTuple{2, T} = (-1.0, 1.0); dmax::Int = 12) where {F, T <: AbstractFloat}
    fun = Fun(f, dom[1] .. dom[2])
    @benchmarkable do_bench($fun, $dmax)
end

function do_bench(fun::Fun, dmax = 12)
    out = 0.0
    for d in dmax:-1:0, m in d:-1:0
        n = d - m
        p, q, λ = rationalcf(fun, m, n)
        out += Float64(λ) # ensure loop body not optimized away
    end
    return out
end

runge(x) = 1 / (1 + 25x^2)
gaussian(x) = exp(-x^2)
skewed_gaussian(x) = x * exp(-x^2)
sinc10(x) = sinc(10x)

const fs = [exp, runge, gaussian, skewed_gaussian, sinc10]
const doms = [(-1.0, 1.0), (-1.0, 2.0)]

const SUITE = BenchmarkGroup()
for f in fs, dom in doms
    SUITE["$(repr(f))"]["$(dom)"] = make_bench(f, dom)
end

if !isinteractive()
    if isfile("params.json")
        BenchmarkTools.loadparams!(SUITE, BenchmarkTools.load("params.json")[1], :evals, :samples)
    else
        BenchmarkTools.tune!(SUITE)
        BenchmarkTools.save("params.json", BenchmarkTools.params(SUITE))
    end
    results = BenchmarkTools.run(SUITE; verbose = true)
    BenchmarkTools.save("results.json", results)
    display(results)
end
