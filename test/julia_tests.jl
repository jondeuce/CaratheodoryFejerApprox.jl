@testset "cheb roots ($T)" for T in [Float32, Float64, BigFloat]
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

@testset "cheb extrema ($T): f = $f, dom = $dom" for T in [Float32, Float64, BigFloat], f in TEST_FUNS, dom in [(-1.0, 1.0), (-0.97, 1.32)]
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
