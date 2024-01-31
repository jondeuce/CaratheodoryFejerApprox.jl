@testset "cheb roots ($T)" for T in [Float32, Float64, Double64, BigFloat]
    fun = ChebFun(T, exp)
    r = chebroots(fun)
    @test isempty(r) # no roots

    fun = ChebFun(T, x -> cospi(10x))
    r = chebroots(fun)
    @test length(r) == 20
    @test r ≈ (-19:2:19) ./ T(20) atol = 25 * eps(T) # endpoints not included

    fun = ChebFun(T, x -> sinpi(10x))
    r = chebroots(fun)
    @test length(r) == 21
    @test r ≈ (-10:10) ./ T(10) atol = 25 * eps(T) # endpoints included
end

@testset "cheb extrema ($T): f = $f, dom = $dom" for T in [Float32, Float64, Double64, BigFloat], f in TEST_FUNS, dom in [(-1.0, 1.0), (-0.97, 1.32)]
    fun = ChebFun(T, f, dom)
    xs = range(T.(dom)...; length = 1001)
    δ = T === Float32 ? 1.0f-4 : 10 * eps(T) # fudge factor

    lo, hi = extrema(f, xs)
    fmin, fmax = chebrange(fun)
    @test lo >= fmin - δ
    @test hi <= fmax + δ

    finf = chebinfnorm(fun)
    @test max(abs(lo), abs(hi)) <= finf + δ
end

@testset "exact polynomialcf ($T): dom = $dom" for T in [Float32, Float64, Double64, BigFloat], dom in [(-1.0, 1.0), (-0.97, 1.32)]
    for m in 0:5
        p = rand_polycoeffs(T, m; type = :mono, annulus = (0.0, 2.0))
        Δm = rand(0:2)
        fun = ChebFun(T, x -> evalpoly(x, p), dom)
        p′, _ = monocoeffs(polynomialcf(fun, m + Δm))
        @test p′[1:m+1] ≈ p
    end
end

@testset "exact rationalcf ($T): dom = $dom" for T in [Float32, Float64, Double64, BigFloat], dom in [(-1.0, 1.0), (-0.97, 1.32)]
    for m in 2:3, n in 2:3
        p = rand_polycoeffs(T, m; type = :mono, annulus = (0.0, 2.0))
        q = rand_polycoeffs(T, n; type = :mono, annulus = (2.0, 3.0))
        p, q = normalize_rational(p, q)
        Δm, Δn = rand(0:2), rand(0:2)
        fun = ChebFun(T, x -> evalrat(x, p, q), dom)
        p′, q′ = monocoeffs(rationalcf(fun, m + Δm, n + Δn))
        @test isapprox(p′[1:m+1], p; rtol = T === Float32 ? 0.01f0 : √(eps(T))) # Float32 seems to be much less accurate
        @test isapprox(q′[1:n+1], q; rtol = T === Float32 ? 0.01f0 : √(eps(T)))
    end
end
