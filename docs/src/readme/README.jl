# # CaratheodoryFejerApprox.jl

# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/CaratheodoryFejerApprox.jl/dev/)
# [![Build Status](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/actions/workflows/CI.yml?query=branch%3Amaster)
# [![Coverage](https://codecov.io/gh/jondeuce/CaratheodoryFejerApprox.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jondeuce/CaratheodoryFejerApprox.jl)
# [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Introduction

# `CaratheodoryFejerApprox.jl` offers robust near-minimax approximation of arbitrary smooth functions using the [Carathéodory-Fejér method](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/tree/master#References). This method approximates real functions with polynomials or rationals by transplanting the problem onto the unit disk in the complex plane, applying the Carathéodory-Fejér theorem, and constructing a near-best approximation from the eigenvalues and eigenvectors of a Hankel matrix containing coefficients from the Chebyshev series expansion of the function.

# Much of the functionality in `CaratheodoryFejerApprox.jl` began as a translation of Matlab code from the fantastic [`Chebfun`](https://github.com/chebfun/chebfun/tree/master) package, specifically [`chebfun/@chebfun/cf.m`](https://github.com/chebfun/chebfun/blob/master/%40chebfun/cf.m). Additionally, internally we make heavy use of the analagous Julia package [`ApproxFun.jl`](https://github.com/JuliaApproximation/ApproxFun.jl).

# ## Package Features

# **Polynomial Approximation**

# ```julia
# polynomialcf(f, m::Int) -> RationalApproximant{Float64}
# polynomialcf(f, dom::NTuple{2, T}, m::Int) -> RationalApproximant{T}
# ```

# Approximate a function `f` with a degree `m` polynomial CF approximant on the interval `dom`. If not specified, `dom` defaults to `(-1.0, 1.0)`.

# **Rational Approximation**

# ```julia
# rationalcf(f, m::Int, n::Int) -> RationalApproximant{Float64}
# rationalcf(f, dom::NTuple{2, T}, m::Int, n::Int) -> RationalApproximant{T}
# ```

# Approximate a function `f` with a type `(m, n)` rational CF approximant on the interval `dom`, where `m` is the numerator degree and `n` is the denominator degree. If not specified, `dom` defaults to `(-1.0, 1.0)`.

# **Minimax Fine-Tuning**

# ```julia
# minimax(f, m::Int, n::Int) -> RationalApproximant{Float64}
# minimax(f, dom::NTuple{2, T}, m::Int, n::Int) -> RationalApproximant{T}
# ```

# Compute the type `(m, n)` CF approximant and then, if necessary, fine-tune the approximant to become a true minimax approximant using the [Remez algorithm](https://en.wikipedia.org/wiki/Remez_algorithm). If not specified, `dom` defaults to `(-1.0, 1.0)`.

# **Polynomial Coefficient Basis**

# ```julia
# chebcoeffs(res::RationalApproximant{T}) -> NTuple{2, Vector{T}}
# monocoeffs(res::RationalApproximant{T}; transplant = true) -> NTuple{2, Vector{T}}
# monocoeffs(res::RationalApproximant{T1}, ::Type{T2} = BigFloat; transplant = true) -> NTuple{2, Vector{T1}}
# ```

#=
Extract polynomial coefficients in the monomial basis via `monocoeffs(res)` or in the Chebyshev basis via `chebcoeffs(res)`. As converting to monomial coefficients can be numerically unstable, optionally pass a higher precision type to `monocoeffs` for intermediate computations.

When `transplant = true` (the default), the monomial coefficients correspond to the original function `f(x)` on the interval `dom`. If `transplant = false`, they correspond to the linearly transplanted function `g(t) = f((x - mid) / rad)` where `mid` and `rad` are the midpoint and radius of `dom` and `-1 <= t <= 1`.

Note that, particularly when `|mid|` is large, it can be much more numerically stable to evaluate the approximant via `evalpoly(t, p)` using the non-transplanted coefficients `p` and `t = (x - mid) / rad`.

The Chebyshev coefficients always correspond to the linearly transplanted function `g(t) = f((x - mid) / rad)` used internally; they are not transplanted to `dom`.
=#

# ## Usage

using CaratheodoryFejerApprox #repl

# ### Polynomial approximant

# Compute a degree 5 polynomial CF approximant to `cos(x)` on the interval `[-0.5, 0.5]`:

res = polynomialcf(cos, (-0.5, 0.5), 5) #repl

#=
A few things to note:
- While we requested a degree 5 approximant, `polynomialcf` automatically recognized that `cos(x)` is an even function and therefore truncated the approximant to degree 4
- The polynomial approximant is displayed in the Chebyshev basis, transplanted to the standard interval `[-1, 1]`. In this case, we see that `cos(x) ≈ p(t)` where `t = 2x`
- Despite being low degree the approximant is quite accurate, with an estimated 6-7 digits of worst-case accuracy

Next, let's extract the coefficients of the rational approximant in the monomial basis:
=#

p, q = monocoeffs(res) #repl

#=
The resulting monomial coefficients are close to (but not equal to) the Taylor expansion of `cos(x)` at the origin: `cos(x) = 1 - x^2/2 + x^4/24 + O(x^6) ≈ 1.0 - 0.5 x^2 + 0.04166 x^4`. This is a common feature of minimax approximants; they are often similar to Taylor series expansions, but adjusted to tradeoff accuracy near the Taylor expansion point for accuracy over the whole interval.
=#

# ### Rational approximant

# We can simlarly compute a type `(4, 4)` rational approximant to `exp(x)` on `[-1, 1]` as follows:

res = rationalcf(exp, 4, 4) #repl

# Note again the low error - approximately 10 digits of accuracy - despite the small degrees of the numerator and denominator.

# ### Minimax approximant

# Now compare the type `(4, 4)` CF approximant with the corresponding type `(4, 4)` minimax approximant:

res = minimax(exp, 4, 4) #repl

# We see that the coefficients are identical, and in fact the error bound is sharp over the whole interval. This is the magic of CF approximants: they are often *very nearly* true minimax approximants.

# ### High-precision rational approximant

# Care has been taken to ensure all internal computations work for generic float types, so we can easily compute a high degree minimax approximant in arbitrary precision:

res = minimax(sin, BigFloat.((-1, 1)), 20, 16) #repl

# Here we have computed a type `(20, 16)` minimax approximation to `sin(x)` on `[-1, 1]` accurate to about 55 digits. Note again that internally `sin(x)` was recognized to be an odd function of `x`, and therefore the numerator degree was truncated from 20 to 19.

# ### Application: implementing the modified Bessel function of the first kind of order zero

#=
To show how minimax approximants may be used in practice, let's derive polynomial approximants for the modified Bessel function of the first kind of order zero `I₀(x)` using `polynomialcf`. We'll compare the approximants to those provided by the [`Bessels.jl` package](https://github.com/JuliaMath/Bessels.jl/blob/e69f0030d6b4f7a73880d693560378e2aec7295c/src/BesselFunctions/besseli.jl#L60).

Following `Bessels.jl`, we'll use the following piecewise approximant for `I₀(x)`:

- For `x < 7.75` we let `I₀(x) = 1 + (x/2)^2 P₁((x/2)^2)`
- For `x ≥ 7.75` we let `I₀(x) = exp(x) P₂(1/x) / sqrt(x)`

where `P₁` and `P₂` are polynomial approximants. Note that `I₀(x)` is an even function, so we need only consider positive `x`.

First, we need the true `I₀(x)`. We will compute this via the integral representation `I₀(x) = (1 / π) ∫_0^π exp(x cos(t)) dt`, which can be evaluated precisely using `QuadGK.jl` with `BigFloat`s:
=#

using QuadGK, Bessels #repl

function besseli0_quadgk(x::BigFloat; expscale = false) #repl-block-start
    I, E = quadgk(BigFloat(0.0), BigFloat(π); order = 21, rtol = 1e-50) do t
        ## I₀(x) diverges exponentially for large x; optionally compute exp(-x) I₀(x) instead
        return expscale ? exp(x * (cos(t) - 1)) : exp(x * cos(t))
    end
    return I / π
end; #repl-block-end

besseli0_quadgk(x; kwargs...) = oftype(x, besseli0_quadgk(BigFloat(x); kwargs...)); #repl

# Let's first check that this matches `Bessels.jl`s implementation:
@assert Bessels.besseli0(0.5) ≈ besseli0_quadgk(0.5; expscale = false) #repl

@assert Bessels.besseli0x(100.0) ≈ besseli0_quadgk(100.0; expscale = true) #repl

# Now, we use `polynomialcf` to compute the polynomial `P₁` for `x < 7.75`:

## We are aiming to find P₁(t) such that I₀(x) = 1 + (x/2)^2 P₁((x/2)^2)
xdom = Double64.((0.0, 7.75)); # compute coefficients in higher precision #repl

tdom = (xdom ./ 2) .^ 2; # t = (x/2)^2 #repl

res = polynomialcf(tdom, 13) do t #repl-block-start
    x = 2√t
    I₀ = besseli0_quadgk(x; expscale = false)
    return t == 0 ? one(t) : (I₀ - 1) / t # P₁(t) = (I₀(x) - 1) / (x/2)^2 = (I₀(x) - 1) / t
end #repl-block-end

# We see that the approximant is accurate to about `eps(Float64)` over the interval. Now, let's compare these coefficients to those from `Bessels.jl`:

const P₁_vec = monocoeffs(res)[1] .|> Float64; # coefficients for P₁(t) #repl

const P₁_tup = (P₁_vec...,); # `evalpoly` is faster with tuples of coefficients #repl

maximum(abs, (P₁_tup .- Bessels.besseli0_small_coefs(Float64)) ./ P₁_tup) # maximum relative difference #repl

# Our derived coefficients are identical. Similarly, let's now use `polynomialcf` to compute the polynomial approximant `P₂(t)` for `x ≥ 7.75`:

## We are aiming to find P₂(t) such that I₀(x) = exp(x) P₂(1/x) / sqrt(x)`
tdom = Double64.((1 / 1e6, 1 / 7.75)); # t = 1/x; domain bounds copied from `Bessels.jl` #repl

res = polynomialcf(tdom, 21) do t #repl-block-start
    x = inv(t)
    I₀ₓ = besseli0_quadgk(x; expscale = true) # exp(-x) * I₀(x)
    return √x * I₀ₓ # P₂(t) = √x * exp(-x) * I₀(x)
end #repl-block-end

# The approximant is again accurate to about `eps(Float64)` over the interval. Let's again compare these coefficients to those from `Bessels.jl`:

const P₂_vec = monocoeffs(res)[1] .|> Float64; # coefficients for P₂(t) #repl

const P₂_tup = (P₂_vec...,); # `evalpoly` is faster with tuples of coefficients #repl

maximum(abs, (P₂_tup .- Bessels.besseli0_med_coefs(Float64)) ./ P₂_tup) # maximum relative difference #repl

#=
Again, we see that the derived coefficients are in good agreement.

Finally, we can now define our implementation of `I₀(x)` using the approximants `P₁` and `P₂`. Modifying the code from [`Bessels.jl`](https://github.com/JuliaMath/Bessels.jl/blob/e69f0030d6b4f7a73880d693560378e2aec7295c/src/BesselFunctions/besseli.jl#L60), we have:
=#

function besseli0_cf(x::Float64) #repl-block-start
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, P₁_tup), 1)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), P₂_tup) / sqrt(x)
        return a * s
    end
end; #repl-block-end

@assert Bessels.besseli0(0.5) ≈ besseli0_cf(0.5) #repl

@assert Bessels.besseli0(100.0) ≈ besseli0_cf(100.0) #repl

#=
# References

### [1] Real Polynomial Chebyshev Approximation by the Carathéodory-Fejér Method [(Gutknecht and Trefethen, 1982)](https://epubs.siam.org/doi/abs/10.1137/0719022)

This foundational work introduces a novel method for near-best approximation of real functions by polynomials. The approach involves transforming the problem onto the unit disk and applying the Carathéodory-Fejér theorem. The resulting approximation is constructed from the principal eigenvalue and eigenvector of a Hankel matrix of Chebyshev coefficients. The method offers high-order agreement with the best approximation, making it of both practical and theoretical importance. This package implements this work in `polynomialcf`.

### [2] The Carathéodory-Fejér Method for Real Rational Approximation [(Trefethen and Gutknecht, 1983)](https://epubs.siam.org/doi/abs/10.1137/0720030)

This work extends the Carathéodory-Fejér method for polynomial minimax approximants to real rational approximation, similarly leveraging eigenvalue analysis of a Hankel matrix of Chebyshev coefficients to achieve this. The rational CF approximants frequently approach the best rational approximation with high accuracy. This package implements this work in `rationalcf`.

### [3] A Robust Implementation of the Carathéodory-Fejér Method for Rational Approximation [(Van Deun and Trefethen, 2011)](https://doi.org/10.1007/s10543-011-0331-7)

This work details a robust implementation of the Carathéodory-Fejér method for both polynomial and rational approximation in Matlab within the Chebfun package ecosystem. `CaratheodoryFejerApprox.jl` is based largely on this reference implementation, providing users with a powerful tool for rational approximation in Julia which additionally works for generic float types beyond `Float64`, such as `BigFloat` and `Double64`.
=#
