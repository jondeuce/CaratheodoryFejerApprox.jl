using Aqua
using CaratheodoryFejerApprox
using Test

using ApproxFun: Chebyshev, Fun, Interval
using LinearAlgebra: norm

rand_uniform(a::T, b::T) where {T} = a + (b - a) * rand(T)
rand_uniform(a::T, b::T, n::Int) where {T} = a .+ (b - a) .* rand(T, n)
rand_uniform(dom::Tuple, args...) = rand_uniform(float.(dom)..., args...)

build_fun(a::AbstractArray, dom::Tuple) = Fun(Chebyshev(Interval(float(eltype(a)).(dom)...)), float(a))
build_fun(dom::Tuple) = a -> build_fun(a, dom)

normalize_rat(p::AbstractVector, q::AbstractVector) = p ./ q[1], q ./ q[1]
normalize_rat((p, q)::Tuple) = normalize_rat(p, q)

@testset "CaratheodoryFejerApprox.jl" begin
    if get(ENV, "CI", "false") == "false"
        @testset "Chebfun" verbose = true begin
            # Only run Chebfun tests if testing locally
            include("chebfun.jl")
        end
    end

    @testset "Code quality (Aqua.jl)" begin
        # Typically causes a lot of false positives with ambiguities and/or unbound args checks;
        # unfortunately have to periodically check this manually
        Aqua.test_all(CaratheodoryFejerApprox; ambiguities = false, unbound_args = true)
    end
end
