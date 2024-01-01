using MATLAB

# Load Chebfun
const CHEBFUN_DIR = joinpath(@__DIR__, "chebfun")
@assert isdir(CHEBFUN_DIR)

mxcall(:addpath, 0, @__DIR__)
mxcall(:addpath, 0, mxcall(:genpath, 1, CHEBFUN_DIR))

mxvec(x::AbstractArray) = vec(x)
mxvec(x::Number) = [x]

# Call "cf" to get the Caratheodory-Fejer approximation
function cf(f::String, dom::Tuple, m::Int, n::Int)
    p, q, s = mxcall(:cfapprox, 3, f, Float64[dom...], Float64(m), Float64(n))
    p, q = mxvec(p), mxvec(q)
    return (; p, q, s)
end

@testset "cf helper" begin
    p, q, s = cf("@(x) exp(x)", (-1, 1), 4, 4)
    @test p ≈ [1.054266374523150, 0.511046272951555, 0.054342198722279, 0.003018257867634, 0.000075824328282]
    @test q ≈ [1.053258353489006, -0.506765278331171, 0.053330086734074, -0.002918926164463, 0.000071733245069]
    @test s ≈ 1.538052539804450e-10
end

@testset "polynomialcf" begin
    for m in 0:6, dom in ((-1, 1), (-1, 2))
        p, _, s = cf("@(x) exp(x)", dom, m, 0)
        p̂, ŝ = polynomialcf(exp, dom, m)
        @test p ≈ p̂
        @test s ≈ ŝ
    end
end
