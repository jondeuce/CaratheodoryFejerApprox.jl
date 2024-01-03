using MATLAB

# Load Chebfun
const CHEBFUN_DIR = joinpath(@__DIR__, "chebfun")
@assert isdir(CHEBFUN_DIR)

mxcall(:addpath, 0, @__DIR__)
mxcall(:addpath, 0, mxcall(:genpath, 1, CHEBFUN_DIR))

mxvec(x::AbstractArray) = vec(x)
mxvec(x::Number) = [x]

# Call "cfcoeffs" to get the Caratheodory-Fejer approximation
function cfcoeffs(f::String, dom::Tuple, m::Int, n::Int)
    p, q, s = mxcall(:cfcoeffs, 3, f, Float64[dom...], Float64(m), Float64(n))
    p, q = mxvec(p), mxvec(q)
    return (; p, q, s)
end

@testset "cfcoeffs" begin
    p, q, s = cfcoeffs("@(x) exp(x)", (-1, 1), 4, 4)
    @test p ≈ [1.054266374523150, 0.511046272951555, 0.054342198722279, 0.003018257867634, 0.000075824328282]
    @test q ≈ [1.053258353489006, -0.506765278331171, 0.053330086734074, -0.002918926164463, 0.000071733245069]
    @test s ≈ 1.538052539804450e-10
end

@testset "polynomialcf" begin
    for m in 0:6, dom in ((-1, 1), (-1, 2))
        f = exp
        p, _, s = cfcoeffs("@(x) exp(x)", dom, m, 0)
        p̂, ŝ = polynomialcf(f, dom, m)
        @test p ≈ p̂ atol = 1e-14
        @test s ≈ ŝ atol = 1e-14 rtol = 1e-14

        P, P̂ = build_fun(dom).((p, p̂))
        xs = rand_uniform(dom, 256)
        @test all(isapprox(P(x), P̂(x); atol = 1e-14) for x in xs)
        @test all(isapprox(f(x), P̂(x); atol = 1.5ŝ) for x in xs)
    end
end

@testset "rationalcf" begin
    for m in 0:6, n in 1:6, dom in ((-1, 1), (-1, 2))
        f = exp
        p, q, s = cfcoeffs("@(x) exp(x)", dom, m, n)
        p̂, q̂, ŝ = rationalcf(f, dom, m, n)

        # Rational approximant computation is less numerically stable, so error tolerances are more lenient for coefficients
        @test p ≈ p̂ atol = 1e-8 rtol = 1e-3
        @test q ≈ q̂ atol = 1e-8 rtol = 1e-3
        @test s ≈ ŝ atol = 1e-14 rtol = 1e-14

        P, Q, P̂, Q̂ = build_fun(dom).((p, q, p̂, q̂))
        R = x -> P(x) / Q(x)
        R̂ = x -> P̂(x) / Q̂(x)

        # Rational approximants as a whole, however, should still match eachother fairly strictly when evaluated
        xs = rand_uniform(dom, 256)
        @test all(isapprox(R(x), R̂(x); atol = 1e-12) for x in xs)
        @test all(isapprox(f(x), R̂(x); atol = max(2ŝ, 1e-14)) for x in xs)
    end
end
