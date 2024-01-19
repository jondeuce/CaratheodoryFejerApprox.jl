using MATLAB

# Load Chebfun
const CHEBFUN_DIR = joinpath(@__DIR__, "chebfun")
@assert isdir(CHEBFUN_DIR)

mxcall(:addpath, 0, @__DIR__)
mxcall(:addpath, 0, mxcall(:genpath, 1, CHEBFUN_DIR))

mxvec(x::AbstractArray) = vec(x)
mxvec(x::Number) = [x]

# Call "chebfuncf" to get the Caratheodory-Fejer approximation
function chebfuncf(f::String, dom::Tuple, m::Int, n::Int; parity::Symbol = :generic)
    p, q, λ = mxcall(:chebfuncf, 3, f, Float64[dom...], Float64(m), Float64(n), string(parity))
    p, q = mxvec(p), mxvec(q)
    return (; p, q, λ)
end

@testset "chebfuncf" begin
    p, q, λ = chebfuncf("@(x) exp(x)", (-1.0, 1.0), 4, 4)
    @test p ≈ [1.054266374523150, 0.511046272951555, 0.054342198722279, 0.003018257867634, 0.000075824328282]
    @test q ≈ [1.053258353489006, -0.506765278331171, 0.053330086734074, -0.002918926164463, 0.000071733245069]
    @test λ ≈ 1.538052539804450e-10
end

gaussian(x) = exp(-x^2)
skewed_gaussian(x) = x * exp(-x^2)
sinc10(x) = sinc(10x)
runge(x) = 1 / (1 + 25x^2)

@testset "polynomialcf: $f" for (f, fstr, fparity) in [
    (exp, "@(x) exp(x)", :generic),
    (gaussian, "@(x) exp(-x.^2)", :even),
    (skewed_gaussian, "@(x) x.*exp(-x.^2)", :odd),
    (sinc10, "@(x) sinc(10.*x)", :even),
    (runge, "@(x) 1./(1+25.*x.^2)", :even),
]
    @testset "m=$m, dom=$dom" for m in 0:8, dom in [(-1.0, 1.0), (-0.97, 1.32)]
        parity = dom[1] == -dom[2] ? fparity : :generic
        p, _, λ = chebfuncf(fstr, dom, m, 0; parity)
        p′, _, _, λ′ = polynomialcf(f, dom, m; parity, vscale = 1.0)

        # Polynomial CF is stable; coefficients should match to high precision
        @test compare_chebcoeffs(p, p′; atol = 1e-14, rtol = 1e-14, parity)
        @test isapprox(λ, λ′; atol = 1e-14, rtol = 1e-14)

        # Compare the approximants to eachother and to the original function
        P, P′ = build_fun(dom).((p, p′))
        xs = rand_uniform(dom, 256)
        @test all(isapprox(P(x), P′(x); atol = 1e-14) for x in xs)
        @test all(isapprox(f(x), P′(x); atol = 1.5 * λ′) for x in xs)
    end
end

@testset "rationalcf: $f" for (f, fstr, fparity) in [
    (exp, "@(x) exp(x)", :generic),
    (gaussian, "@(x) exp(-x.^2)", :even),
    (skewed_gaussian, "@(x) x.*exp(-x.^2)", :odd),
]
    @testset "m=$m, n=$n, dom=$dom" for m in 0:6, n in 1:6, dom in ((-1.0, 1.0), (-0.97, 1.32))
        parity = dom[1] == -dom[2] ? fparity : :generic
        p_parity = parity === :even ? :even : parity === :odd ? :odd : :generic
        q_parity = (parity === :even || parity === :odd) ? :even : :generic
        p, q, λ = chebfuncf(fstr, dom, m, n; parity)
        p′, q′, _, λ′ = rationalcf(f, dom, m, n; parity, vscale = 1.0)

        p, q = normalize_rational(p, q)
        p′, q′ = normalize_rational(p′, q′)

        # If the approximant is too close to exact, then the coefficients are not
        # not unique in machine precision, and we will get spurious failures,
        # so only compare if the error is large enough
        if max(abs(λ), abs(λ′)) > 10 * eps()
            # Rational approximant computation is less numerically stable, so error tolerances are more lenient for coefficients
            @test compare_chebcoeffs(p, p′; atol = 1e-8, rtol = 1e-4, parity = p_parity)
            @test compare_chebcoeffs(q, q′; atol = 1e-8, rtol = 1e-4, parity = q_parity)
            @test isapprox(λ, λ′; atol = 1e-14, rtol = 1e-14)
        end

        P, Q, P′, Q′ = build_fun(dom).((p, q, p′, q′))
        R = x -> P(x) / Q(x)
        R′ = x -> P′(x) / Q′(x)

        # Compare the approximants to eachother and to the original function
        xs = rand_uniform(dom, 256)
        @test maximum(abs, R.(xs) .- R′.(xs)) <= max(1e-8, 1e-4 * maximum(abs, R.(xs)))
        @test maximum(abs, f.(xs) .- R′.(xs)) <= max(2 * λ′, 1e-14)
    end
end
