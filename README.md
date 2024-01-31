# CaratheodoryFejerApprox.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jondeuce.github.io/CaratheodoryFejerApprox.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jondeuce.github.io/CaratheodoryFejerApprox.jl/dev/)
[![Build Status](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jondeuce/CaratheodoryFejerApprox.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jondeuce/CaratheodoryFejerApprox.jl)

## Introduction

`CaratheodoryFejerApprox.jl` offers robust near-minimax approximation of arbitrary smooth functions using the [Carathéodory-Fejér method](https://github.com/jondeuce/CaratheodoryFejerApprox.jl/tree/master#References). This method provides an efficient way to approximate real functions with polynomials or rationals by transplanting the problem to the unit disk in the complex plane, applying the Carathéodory-Fejér theorem, and constructing a near-best approximation from the eigenvalues and eigenvectors of a Hankel matrix containing coefficients from the Chebyshev series expansion of the function.

Much of the functionality in `CaratheodoryFejerApprox.jl` began as a translation of Matlab code from the fantastic [`Chebfun`](https://github.com/chebfun/chebfun/tree/master) package, specifically [`chebfun/@chebfun/cf.m`](https://github.com/chebfun/chebfun/blob/master/%40chebfun/cf.m). Additionally, internally we make heavy use of the analagous Julia package [`ApproxFun.jl`](https://github.com/JuliaApproximation/ApproxFun.jl).

## Package Features

**Polynomial Approximation**

```julia
polynomialcf(f, m::Int) -> RationalApproximant{Float64}
polynomialcf(f, dom::NTuple{2, T}, m::Int) -> RationalApproximant{T}
```
Approximate a function `f` with a degree `m` polynomial CF approximant on the interval `dom`. If not specified, `dom` defaults to `(-1.0, 1.0)`.

**Rational Approximation**

```julia
rationalcf(f, m::Int, n::Int) -> RationalApproximant{Float64}
rationalcf(f, dom::NTuple{2, T}, m::Int, n::Int) -> RationalApproximant{T}
```
Approximate a function `f` with a type `(m, n)` rational CF approximant on the interval `dom`, where `m` is the numerator degree and `n` is the denominator degree. If not specified, `dom` defaults to `(-1.0, 1.0)`.

**Minimax Fine-Tuning**

```julia
minimax(f, m::Int, n::Int) -> RationalApproximant{Float64}
minimax(f, dom::NTuple{2, T}, m::Int, n::Int) -> RationalApproximant{T}
```
Compute the type `(m, n)` CF approximant and then, if necessary, fine-tune the approximant to become a true minimax approximant using the [Remez algorithm](https://en.wikipedia.org/wiki/Remez_algorithm). If not specified, `dom` defaults to `(-1.0, 1.0)`.

**Coefficient Basis**

```julia
chebcoeffs(res::RationalApproximant{T}) -> NTuple{2, Vector{T}}
monocoeffs(res::RationalApproximant{T}; transplant = true) -> NTuple{2, Vector{T}}
monocoeffs(res::RationalApproximant{T1}, ::Type{T2} = BigFloat; transplant = true) -> NTuple{2, Vector{T1}}
```
Extract coefficients in the monomial basis via `monocoeffs(res)` or in the Chebyshev basis via `chebcoeffs(res)`. As converting to monomial coefficients can be numerically unstable, optionally pass a higher precision type to `monocoeffs` for intermediate computations.

When `transplant = true` (the default), the monomial coefficients correspond to the original function `f(x)` on the interval `dom`. If `transplant = false`, they correspond to the linearly transplanted function `f(t) = f((x - mid) / rad)` where `-1 <= t <= 1`, `mid` is the midpoint `(dom[1] + dom[2]) / 2`, and `rad` is the radius `(dom[2] - dom[1]) / 2`. Note that, particularly when `dom` is large, it can be much more numerically stable to evaluate the approximant via `evalpoly(t, p)` where `p` are the non-transplanted coefficients and `t = (x - mid) / rad` is the rescaled input variable.

The Chebyshev coefficients always correspond to the linearly transplanted function `f(t) = f((x - mid) / rad)` used internally, i.e. `transplant = false` above.

## Usage

Compute a degree 5 polynomial CF approximant to `cos(x)` on the interval `[-0.5, 0.5]`:

```julia
julia> res = polynomialcf(cos, (-0.5, 0.5), 5)
RationalApproximant{Float64}
  Type:   m / n = 4 / 0
  Domain: -0.5 ≤ x ≤ 0.5
  Error:  |f(x) - p(x)| ⪅ 6.721e-7
  Approximant:
      p(t) = 0.9385⋅T_0(t) - 0.06121⋅T_2(t) + 0.0003215⋅T_4(t)
  where: t = (x - 0.0) / 0.5
```

Then, extract the coefficients of the rational approximant in the monomial basis:

```julia
julia> p, q = monocoeffs(res)
([0.9999993278622336, 0.0, -0.49995153387633173, 0.0, 0.04114863415981116], [1.0])
```

A few things to note:
- While we requested a degree 5 approximant, `polynomialcf` automatically recognized that `cos(x)` is an even function and therefore truncated the approximant to degree 4
- The approximant, despite being quite low degree, is a rather accurate: an estimated 6-7 digits of worst-case accuracy
- The resulting monomial coefficients are close to (though not equal to) the Taylor expansion of `cos(x)` at `x=0`: `cos(x) = 1 - x^2/2 + x^4/24 + O(x^6) ≈ 1.0 - 0.5 x^2 + 0.04166 x^4`. This is a common feature of minimax approximants; they are less accurate than Taylor series near the point of expansion, but more accurate over the whole interval.

Simlarly, we can compute a type `(4, 4)` rational approximant to `exp(x)` on `[-1, 1]` as follows:

```julia
julia> res = rationalcf(exp, 4, 4)
RationalApproximant{Float64}
  Type:   m / n = 4 / 4
  Domain: -1.0 ≤ x ≤ 1.0
  Error:  |f(x) - p(x) / q(x)| ⪅ 1.538e-10
  Approximant:
      p(x) = 1.054⋅T_0(x) + 0.511⋅T_1(x) + 0.05434⋅T_2(x) + 0.003018⋅T_3(x) + 7.582e-5⋅T_4(x)
      q(x) = 1.053⋅T_0(x) - 0.5068⋅T_1(x) + 0.05333⋅T_2(x) - 0.002919⋅T_3(x) + 7.173e-5⋅T_4(x)
```

Note again the high degree of accuracy despite the low degrees of the numerator and denominator.

Now compare the type `(4, 4)` CF approximant with the corresponding type `(4, 4)` minimax approximant:

```julia
julia> res = minimax(exp, 4, 4)
RationalApproximant{Float64}
  Type:   m / n = 4 / 4
  Domain: -1.0 ≤ x ≤ 1.0
  Error:  |f(x) - p(x) / q(x)| ⪅ 1.538e-10
  Approximant:
      p(x) = 1.054⋅T_0(x) + 0.511⋅T_1(x) + 0.05434⋅T_2(x) + 0.003018⋅T_3(x) + 7.582e-5⋅T_4(x)
      q(x) = 1.053⋅T_0(x) - 0.5068⋅T_1(x) + 0.05333⋅T_2(x) - 0.002919⋅T_3(x) + 7.173e-5⋅T_4(x)
```

We see that the coefficients are identical, and in fact the error bound is sharp over the whole interval. This is the magic of CF approximants: they are often *very nearly* true minimax approximants.

Care has been taken to ensure all internal computations work for generic types, so we can easily compute a high degree minimax approximant in arbitrary precision, as well:

```julia
julia> res = minimax(sin, BigFloat.((-1, 1)), 20, 16)
RationalApproximant{BigFloat}
  Type:   m / n = 19 / 16
  Domain: -1.0 ≤ x ≤ 1.0
  Error:  |f(x) - p(x) / q(x)| ⪅ 8.282e-56
  Approximant:
      p(x) = 4.676e-62⋅T_0(x) + 0.8859⋅T_1(x) + 4.48e-62⋅T_2(x) + ...
      q(x) = 1.004⋅T_0(x) + 1.063e-61⋅T_1(x) + 0.00447⋅T_2(x) + ...
```

Here we have computed a type `(20, 16)` minimax approximation to `sin(x)` on `[-1, 1]` accurate to about 55 digits. Note again that internally `sin(x)` was recognized to be an odd function of `x`, and therefore the numerator degree was truncated from 20 to 19.

# References

### [1] Real Polynomial Chebyshev Approximation by the Carathéodory-Fejér Method
[Gutknecht and Trefethen (1982)](https://epubs.siam.org/doi/abs/10.1137/0719022)

This foundational work introduces a novel method for near-best approximation of real functions by polynomials. The approach involves transforming the problem onto the unit disk and applying the Carathéodory-Fejér theorem. The resulting approximation is constructed from the principal eigenvalue and eigenvector of a Hankel matrix of Chebyshev coefficients. The method offers high-order agreement with the best approximation, making it of both practical and theoretical importance. This package implements this technique for achieving robust near-minimax approximation via the `polynomialcf` function.

### [2] The Carathéodory-Fejér Method for Real Rational Approximation
[Trefethen and Gutknecht (1983)](https://epubs.siam.org/doi/abs/10.1137/0720030)

This work extends the Carathéodory-Fejér method for polynomial minimax approximants to real rational approximation, similarly leveraging eigenvalue analysis of a Hankel matrix of Chebyshev coefficients to achieve this. The rational CF approximants frequently approach the best rational approximation with high accuracy.

### [3] A Robust Implementation of the Carathéodory-Fejér Method for Rational Approximation
[Van Deun and Trefethen (2011)](https://doi.org/10.1007/s10543-011-0331-7)

This work focuses on a robust implementation of the Carathéodory-Fejér method for both polynomial and rational approximation in Matlab within the Chebfun package ecosystem. `CaratheodoryFejerApprox.jl` is based largely on this reference implementation, providing users with a powerful tool for robust approximation in the Julia programming language which additionally works for generic types beyond `Float64` such as `BigFloat` and `Double64`.
