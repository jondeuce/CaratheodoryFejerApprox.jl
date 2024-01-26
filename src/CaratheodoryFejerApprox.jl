#=
CaratheodoryFejerApprox.jl is based largely on the following papers:

    [1] Gutknecht MH, Trefethen LN. Real Polynomial Chebyshev Approximation by the Carathéodory–Fejér method. SIAM J Numer Anal 1982; 19: 358–371.
    [2] Trefethen LN, Gutknecht MH. The Carathéodory–Fejér Method for Real Rational Approximation. SIAM J Numer Anal 1983; 20: 420–436.
    [3] Henrici P. Fast Fourier Methods in Computational Complex Analysis. SIAM Rev 1979; 21: 481–527.

As well as the following files from Chebfun v5:

    chebfun/@chebfun/cf.m
    chebfun/@chebfun/chebpade.m
    chebfun/@chebtech/roots.m

Chebfun v5 is distributed under the following license:

Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
of Oxford, and the Chebfun Developers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Oxford nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

module CaratheodoryFejerApprox

using AbstractFFTs: AbstractFFTs, Plan, irfft, plan_rfft, rfft
using ApproxFun: ApproxFun, Chebyshev, ChebyshevInterval, ClosedInterval, Fun, Interval, coefficients, domain, endpoints, extrapolate, ncoefficients, space
using ArnoldiMethod: ArnoldiMethod, partialschur
using Arpack: Arpack, eigs
using DoubleFloats: DoubleFloats, Double64
using FFTW: FFTW # defines fast FFTs for real/complex Float32/Floa64
using GenericFFT: GenericFFT # defines generic methods for FFTs
using GenericLinearAlgebra: GenericLinearAlgebra # defines generic method for `LinearAlgebra.eigen` and `Polynomials.roots`
using LinearAlgebra: LinearAlgebra, Hermitian, cholesky!, cond, diag, dot, eigen, eigvals, ldiv!, mul!, qr!
using Polynomials: Polynomials, ChebyshevT, Polynomial
using PrecompileTools: PrecompileTools, @compile_workload
using Roots: Roots
using Setfield: Setfield, @set!
using ToeplitzMatrices: ToeplitzMatrices, Toeplitz, Hankel

export Double64 # re-export
export minimax, polynomialcf, rationalcf
export chebcoeffs, monocoeffs

struct RationalApproximant{T <: AbstractFloat}
    p::Vector{T}
    q::Vector{T}
    dom::NTuple{2, T}
    err::T
end
PolynomialApproximant(p::AbstractVector{T}, dom::NTuple{2, T}, err::T) where {T <: AbstractFloat} = RationalApproximant(p, ones(T, 1), dom, err)

function (res::RationalApproximant{T})(x) where {T <: AbstractFloat}
    x = T(x)
    pₓ = ChebFun(T, res.p, res.dom)(x)
    if length(res.q) <= 1
        return pₓ
    else
        qₓ = ChebFun(T, res.q, res.dom)(x)
        return pₓ / qₓ
    end
end

Base.Tuple(rat::RationalApproximant) = (rat.p, rat.q, rat.dom, rat.err)
Base.NamedTuple(rat::RationalApproximant) = (; rat.p, rat.q, rat.dom, rat.err)
Base.iterate(rat::RationalApproximant, state...) = iterate(Tuple(rat), state...)

function monocoeffs(rat::RationalApproximant{T1}, ::Type{T2} = BigFloat; transplant = true) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
    p′ = poly(T2.(rat.p), T2.(rat.dom); transplant)
    q′ = poly(T2.(rat.q), T2.(rat.dom); transplant)
    p′, q′ = normalize_rational!(p′, q′)
    return T1.(p′), T1.(q′)
end
chebcoeffs(rat::RationalApproximant) = (rat.p, rat.q)

function Base.show(io::IO, rat::RationalApproximant{T}) where {T <: AbstractFloat}
    (; p, q, dom, err) = rat
    m, n = length(p) - 1, length(q) - 1
    rnd = x -> round(Float64(x); sigdigits = 4)
    is_unit = dom == (-1, 1)
    var = is_unit ? :x : :t
    μ, σ = is_unit ? (zero(T), one(T)) : (rnd((dom[1] + dom[2]) / 2), rnd((dom[2] - dom[1]) / 2))
    num, den = ChebyshevT{Float64, var}(rnd.(p)), ChebyshevT{Float64, var}(rnd.(q))
    println(io, "RationalApproximant{", T, "}")
    println(io, "  Type:   m / n = ", m, " / ", n)
    println(io, "  Domain: ", rnd(dom[1]), " ≤ x ≤ ", rnd(dom[2]))
    println(io, "  Error:  |f(x) - ", (n == 0 ? "p(x)" : "p(x) / q(x)"), "| ⪅ ", rnd(err))
    println(io, "  Approximant:")
    print("      p(", var, ") = ", num)
    n > 0 && print("\n      q(", var, ") = ", den)
    !is_unit && print("\n  where: t = (x - ", μ, ") / ", σ)
    return nothing
end

Base.@kwdef struct CFOptions{T <: AbstractFloat}
    "Domain of f(x)"
    dom::NTuple{2, T} = (-one(T), one(T))
    "Even/odd parity of f(x)"
    parity::Symbol = :generic
    "Upper bound on the maximum value of |f(x)|"
    vscale::T = one(T)
    "Maximum Chebyshev series degree. Series is truncated to this degree if necessary"
    maxdegree::Int = 2^10
    "Relative tolerance for chopping off small coefficients"
    rtolchop::T = eps(T)
    "Absolute tolerance for chopping off small coefficients"
    atolchop::T = zero(T)
    "Absolute tolerance for detecting rationality"
    atolrat::T = 50 * eps(T)
    "Relative tolerance for Fourier integrals"
    rtolfft::T = 50 * eps(T)
    "Minimum FFT length"
    minnfft::Int = 2^8
    "Maximum FFT length"
    maxnfft::Int = 2^20
    "Relative tolerance for detecting ill-conditioning"
    rtolcond::T = 1 / (500 * eps(T))
    "Absolute tolerance for detecting ill-conditioning"
    atolcond::T = T(1000)
    "Suppress warnings"
    quiet::Bool = false
end
const CFOptionsFieldnames = Symbol[fieldnames(CFOptions)...]

CFOptions(c::AbstractVector{T}, dom::Tuple; kwargs...) where {T <: AbstractFloat} = CFOptions{T}(; dom = check_endpoints(T.(dom)), parity = parity(c), vscale = vscale(c), kwargs...)
CFOptions(o::CFOptions{T}, c::AbstractVector{T}) where {T <: AbstractFloat} = (@set! o.parity = parity(c); @set! o.vscale = vscale(c); return o)

polynomialcf(f, m::Int; kwargs...) = polynomialcf(ChebFun(f), m; kwargs...)
polynomialcf(f, dom::Tuple, m::Int; kwargs...) = polynomialcf(ChebFun(f, dom), m; kwargs...)
polynomialcf(fun::Fun, m::Int; kwargs...) = polynomialcf(coefficients_and_endpoints(fun)..., m; kwargs...)
polynomialcf(c::AbstractVector{T}, m::Int; kwargs...) where {T <: AbstractFloat} = polynomialcf(c, (-one(T), one(T)), m; kwargs...)
polynomialcf(c::AbstractVector{T}, dom::Tuple, m::Int; kwargs...) where {T <: AbstractFloat} = polynomialcf(c, m, CFOptions(c, dom; kwargs...))

rationalcf(f, m::Int, n::Int; kwargs...) = rationalcf(ChebFun(f), m, n; kwargs...)
rationalcf(f, dom::Tuple, m::Int, n::Int; kwargs...) = rationalcf(ChebFun(f, dom), m, n; kwargs...)
rationalcf(fun::Fun, m::Int, n::Int; kwargs...) = rationalcf(coefficients_and_endpoints(fun)..., m, n; kwargs...)
rationalcf(c::AbstractVector{T}, m::Int, n::Int; kwargs...) where {T <: AbstractFloat} = rationalcf(c, (-one(T), one(T)), m, n; kwargs...)
rationalcf(c::AbstractVector{T}, dom::Tuple, m::Int, n::Int; kwargs...) where {T <: AbstractFloat} = rationalcf(c, m, n, CFOptions(c, dom; kwargs...))

function polynomialcf(c::AbstractVector{T}, m::Int, o::CFOptions) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested polynomial degree m = $(m) must be non-negative"

    iszero(o.vscale) && return postprocess(zeros(T, 1), zero(T), o) # trivial function
    c = c[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    c ./= o.vscale # normalize scale such that max|f(x)| ≈ 1
    c = lazychopcoeffs(c; rtol = o.rtolchop, atol = o.atolchop) # view of non-negligible coefficients
    M = length(c) - 1
    m = min(m, M)

    # Check even/odd symmetries
    if o.parity === :even
        m -= isodd(m) # ensure m is even
        isodd(M) && (M -= 1; c = @views c[1:end-1]) # ensure M is even
    elseif o.parity === :odd
        m > 0 && (m -= iseven(m)) # ensure m is odd, or zero
        M > 0 && iseven(M) && (M -= 1; c = @views c[1:end-1]) # ensure M is odd, or zero
    end

    # Trivial cases
    (m == M) && return postprocess(c[1:M+1], zero(T), o)
    (m == M - 1) && return postprocess(c[1:M], abs(c[M+1]), o)
    (m == 0) && return polynomialcf_constant(c, o)

    if o.parity === :even
        # If f(x) is even, we can reduce the problem to polynomial approximation of g(x) = f(T₂(x))
        c′ = @views c[1:2:end]
        p′, _, _, λ = polynomialcf(c′, m ÷ 2, CFOptions(o, c′))
        p = unzip(p′; even = true)
        return postprocess(p, λ, o)
    end

    cm = @views c[m+2:M+1]
    D, V = eigs_hankel(cm; nev = 1)
    λ, u = only(D), vec(V)

    u1 = u[1]
    uu = @views u[2:M-m]
    b = [zeros(T, 2m + 1); cm]
    for k in -m:m
        b[m-k+1] = @views -dot(b[m-k+2:M-k], uu) / u1
    end
    @views b[m+2:2m+1] .+= b[m:-1:1]
    p = @views c[1:m+1] - b[m+1:2m+1]

    return postprocess(p, abs(λ), o)
end

function polynomialcf_constant(c::AbstractVector{T}, o::CFOptions{T}) where {T <: AbstractFloat}
    M = length(c) - 1
    D, V = @views eigs_hankel(c[2:M+1]; nev = 1)
    λ, u = only(D), vec(V)
    p = @views c[1] + dot(c[2:M], u[2:M]) / u[1]
    return postprocess([p], abs(λ), o)
end

function rationalcf(c::AbstractVector{T}, m::Int, n::Int, o::CFOptions{T}) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested numerator degree m = $(m) must be non-negative"
    @assert 0 <= n "Requested denominator degree n = $(n) must be non-negative"

    iszero(o.vscale) && return postprocess(zeros(T, 1), ones(T, 1), zero(T), o) # trivial function
    c = c[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    c ./= o.vscale # normalize scale such that max|f(x)| ≈ 1
    c = lazychopcoeffs(c; rtol = o.rtolchop, atol = o.atolchop) # view of non-negligible coefficients
    M = length(c) - 1
    m = min(m, M)

    # For even/odd symmetric functions, we first adjust (m, n) to respect the function symmetry
    if o.parity === :even
        m -= isodd(m) # ensure m is even
        n -= isodd(n) # ensure n is even
        isodd(M) && (M -= 1; c = @views c[1:end-1]) # ensure M is even
    elseif o.parity === :odd
        m > 0 && (m -= iseven(m)) # ensure m is odd, or zero
        n -= isodd(n) # ensure n is even
        M > 0 && iseven(M) && (M -= 1; c = @views c[1:end-1]) # ensure M is odd, or zero
    end

    # Trivial cases
    (n == 0 || m == M) && return rationalcf_reduced(c, m, o)
    (m == 0 && o.parity === :odd) && return rationalcf_reduced(c, 0, o) # best approximation is constant

    # Check even/odd symmetries
    if o.parity === :even
        # If f(x) is even, we can reduce the problem to rational approximation of g(x) = f(T₂(x))
        c′ = @views c[1:2:end]
        p′, q′, _, λ = rationalcf(c′, m ÷ 2, n ÷ 2, CFOptions(o, c′))
        p = unzip(p′; even = true)
        q = unzip(q′; even = true)
        return postprocess(p, q, λ, o)
    end

    # Now, for even/odd symmetric functions, although we adjusted (m, n) above to respect the function symmetry,
    # the CF operator is continuous (and numerically stable) if and only if (m, n) lies in the lower-left or upper-right
    # corner of a 2x2 square block in the CF table. We increase m accordingly, and later prune the extra zero in the numerator.
    if o.parity === :even
        m += 1 # move to upper right corner: (m, n) -> (m + 1, n) (note: m and n are both even)
    elseif o.parity === :odd
        isodd(m) && (m += 1) # move to lower left corner: (m, n) -> (m + 1, n) (note: m is odd or zero, and n is even)
    end

    # Reorder coeffs and scale T_0 coefficient
    a = copy(c)
    a[1] *= 2
    reverse!(a)

    # Obtain eigenvalues and block structure
    λ, u, k, l, is_rat = eigs_hankel_block(a, m, n; atol = o.atolrat)
    if k > 0 || l > 0
        if is_rat
            # f is rational (at least up to machine precision)
            p, q = pade(c, m - k, n - k)
            return postprocess(p, q, eps(T), o)
        end

        # Try upper-left (m + l, n - k) corner of CF table block
        λ′, u′, k′, l′, _ = eigs_hankel_block(a, m + l, n - k; atol = o.atolrat)
        if k′ > 0 || l′ > 0
            # Otherwise, go to lower-right (m - k, n + l) corner of CF table block
            λ, u, _, _, _ = eigs_hankel_block(a, m - k, n + l; atol = o.atolrat)
            n = n + l
        else
            λ, u = λ′, u′
            n = n - k
        end
    end

    # Compute Chebyshev coefficients of approximation R̃(x) (eqn. 1.7b[2]) from the Laurent coefficients
    # of the Blaschke product b(z) (eqn. 1.6b[2]) via FFT
    N = max(nextpow(2, length(u)), o.minnfft)
    c̃, c̃_prev, Δc̃ = zeros(T, m + 1), zeros(T, m + 1), T(Inf) # Chebyshev coefficients of R̃(x): [c̃₀, c̃₁, ..., c̃ₘ] = [2*a₀, a₁ + a₋₁, ..., aₘ + a₋ₘ]
    while true
        c̃_prev, c̃ = c̃, c̃_prev
        c̃_full = blaschke_laurent_coeffs(u, N, M) # Laurent coefficients of R̃(x): [a₀, a₁, ..., aₖ₋₁, a₋ₖ, ..., a₋₁] where k = N ÷ 2
        @views c̃ .= c̃_full[1:m+1]
        @views c̃[2:end] .+= c̃_full[end:-1:end-m+1]
        c̃[1] *= 2
        Δc̃_prev, Δc̃ = Δc̃, maximum(abs, c̃ - c̃_prev)
        ((N *= 2) > o.maxnfft || (Δc̃ < √o.rtolfft * maximum(abs, c̃) && Δc̃ > Δc̃_prev / 2) || isapprox(c̃, c̃_prev; rtol = o.rtolfft)) && break
    end
    @views c̃ .= a[end:-1:end-m] .- λ .* c̃ # Chebyshev coefficients of R̃(x) (eqn. 1.7b[2])
    λ = abs(λ)

    # Compute the polynomial q₁(z) via incomplete factorization (sec. 3.2[3]) of the denominator of the Blaschke product b(z) (eqn. 1.6b[2]).
    # This is done by computing the Laurent coefficients of p'(z) / p(z) via FFT, where p(z) is the numerator of b(z),
    # noting that the numerator of b(z) is the reciprocal polynomial zⁿ⋅b(z⁻¹) of the denominator.
    N = max(nextpow(2, length(u)), o.minnfft)
    ∇u = polyder(u)
    s̃, s̃_prev, Δs̃ = zeros(T, n), zeros(T, n), T(Inf)
    while true
        s̃_prev, s̃ = s̃, s̃_prev
        @views s̃ .= incomplete_factorization_laurent_coeffs(u, ∇u, N)[end-n:end-1]
        Δs̃_prev, Δs̃ = Δs̃, maximum(abs, s̃ - s̃_prev)
        ((N *= 2) > o.maxnfft || (Δs̃ < √o.rtolfft * maximum(abs, s̃) && Δs̃ > Δs̃_prev / 2) || isapprox(s̃, s̃_prev; rtol = o.rtolfft)) && break
    end

    # Compute coefficients of the incomplete factorization q₁(z) = ∑ᵢ₌₀ⁿ bᵢzⁱ (eqn. 3.11[3])
    b = ones(T, n + 1)
    for j in 1:n
        b[j+1] = @views -dot(b[1:j], s̃[end-j+1:end]) / j
    end
    reverse!(b)

    # Compute the Chebyshev coefficients of the polynomial Q(x) = q₁(z)q₁(z⁻¹)/τ = |q₁(z)|²/τ (eqn. 1.10[2]) where z ∈ ∂D⁺ and τ = q₁(i)q₁(-i).
    q, ρ = normalized_symmetric_laurent_product(b; o.quiet)
    n = length(q) - 1 # denominator degree may change in degenerate cases
    Q = Fun(ChebSpace(T), q)

    # Compute numerator polynomial from Chebyshev expansion of 1/Q(x) and R̃(x)
    if n > 0
        # We know the exact ellipse of analyticity for 1/Q(x), so use this knowledge to obtain its Chebyshev coefficients:
        ρ⁻¹ = inv(clamp(ρ, eps(T), 1 - eps(T)))
        nq⁻¹ = ceil(Int, log(4 / eps(T) / (ρ⁻¹ - 1)) / log(ρ⁻¹))
        Q⁻¹ = Fun(x -> inv(Q(x)), ChebSpace(T), min(nq⁻¹, o.maxdegree))
        γ = zeropadresize!(coefficients(Q⁻¹), 2m + 1)
    else
        γ = zeros(T, 2m + 1)
        γ[1] = inv(q[1])
    end
    γ₀ = γ[1] *= 2
    Γ = toeplitz(γ)

    if m == 0
        p = [c̃[1] / γ₀]
        return postprocess(p, q, λ, o)
    end

    # The following steps reduce the Toeplitz system of size 2*m + 1 to a system of
    # size m, and then solve it. If q has zeros close to the domain, then G is
    # ill-conditioned, and accuracy is lost
    A = @views Γ[1:m, 1:m]
    B = @views Γ[1:m, m+1]
    C = @views Γ[1:m, end:-1:m+2]
    G = Hermitian(A .+ C .- (2 / γ₀) .* B .* B')

    if cond(G) > max(λ * o.rtolcond, o.atolcond)
        !o.quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
    end

    p = zeros(T, m + 1)
    @views ldiv!(p[1:m], LinearAlgebra.cholesky!(G), (-2 * c̃[1] / γ₀) .* B .+ 2 .* c̃[m+1:-1:2])
    @views p[m+1] = (c̃[1] - dot(p[1:m], B)) / γ₀
    reverse!(p)

    return postprocess(p, q, λ, o)
end

function rationalcf_reduced(c::AbstractVector, m::Int, o::CFOptions)
    p, q, _, λ = polynomialcf(c, m, CFOptions(o, c))
    return postprocess(p, q, λ, o)
end

function normalized_symmetric_laurent_product(b::AbstractVector{T}; quiet::Bool = false) where {T <: AbstractFloat}
    # Compute the Chebyshev coefficients of the polynomial Q(x) = q(z)q(z⁻¹)/τ = |q(z)|²/τ (eqn. 1.10[2]),
    # where z ∈ ∂D⁺, τ = q(i)q(-i), and q(z) = ∑ᵢ₌₀ⁿ ẽᵢzⁱ denotes "the normalized denominator of r̃∗(z) (eqn. 1.6a[2]) -
    # the polynomial of degree <= n with constant term ẽ₀ = 1 whose zeros are the finite poles of r̃∗(z) lying outside ∂D"[2].
    # Note: we work with the reciprocal polynomial q₁(z) = zⁿ⋅q(z⁻¹) = ∑ᵢ₌₀ⁿ bᵢzⁱ, which satisfies q(z)q(z⁻¹) = q₁(z)⋅q₁(z⁻¹)
    # and has zeros inside ∂D. Returns Chebyshev coefficients of Q(x) and the radius ρ of the zero-free annulus ρ < |z| < ρ⁻¹.
    z = polyroots(b)
    ρ = maximum(abs, z)

    if ρ < 1
        # Normal case; roots are all within the unit disk. Compute Chebyshev coefficients of Q(x) by expanding the product directly:
        #   Q̃(x) = q₁(z)q₁(z⁻¹) = (∑ᵢ₌₀ⁿ bᵢ zⁱ) (∑ⱼ₌₀ⁿ bⱼ z⁻ʲ) = ∑ᵢ,ⱼ₌₀ⁿ bᵢ bⱼ zⁱ⁻ʲ
        #        = ∑′ₖ₌₀ⁿ ∑ₗ₌ₖⁿ bₖ bₗ (zᵏ + z⁻ᵏ)    where ∑′ means k=0 term is halved
        #        = ∑′ₖ₌₀ⁿ (2 bₖ ∑ₗ₌ₖⁿ bₗ) Tₖ(x)     where Tₖ(x) = (zᵏ + z⁻ᵏ) / 2 for z ∈ ∂D⁺
        n = length(b) - 1
        q = zeros(T, n + 1)
        for k in 1:n+1
            q[1] += b[k]^2
            for l in k+1:n+1
                q[l-k+1] += 2 * b[k] * b[l]
            end
        end
        normalize_chebpoly!(q) # normalize by τ = q₁(i)q₁(-i) = Q̃(0)
    else
        # Degenerate case; roots are not all within the unit disk. As in chebfun, we filter out roots with |z| > 1
        # and then compute Q(x) using the renormalized q₁(z).
        !quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
        filter!(zi -> abs(zi) < 1, z)
        isempty(z) && return ones(T, 1), zero(T)
        ρ = maximum(abs, z)
        n = length(z)

        # Form Q(x) using Chebyshev interpolation of q₁(z)q₁(z⁻¹)/τ; this is more numerically stable than expanding ∏ᵢ₌₁ⁿ (z - zᵢ)(z⁻¹ - zᵢ)
        @. z = (z + inv(z)) / 2
        Π = prod(-, z)
        Q = Fun(x -> real(prod(zi -> x - zi, z) / Π), ChebSpace(T), n + 1)
        q = coefficients(Q)
    end

    return q, ρ
end

function blaschke_laurent_coeffs(u::AbstractVector{T}, N::Int, M::Int) where {T <: AbstractFloat}
    # Compute Laurent coefficients of the quotient of Blaschke products:
    #   b(z) = z^M * P(z) / Q(z)
    #        = z^M * (∑_{k=0}^{d-1} u_k z^k) / (∑_{k=0}^{d-1} u_{d-k-1} z^k).
    # To efficiently evaluate b(z) for many z on the unit circle, u is padded u with
    # zeros to length N and the numerator is computed using the FFT. Additionally,
    # we save an FFT by using shift/reflect FFT identities to compute the denominator.
    d = length(u)
    θ = (2 * (M - d + 1) // N) * (T(0):T(N ÷ 2)) # denominator shifts
    u = zeropad(u, N)
    p = get_plan_rfft(u)
    û = p * u
    @. û = cispi(-θ) * z_div_conj_z(û)
    return ldiv!(u, p, û) # output is equivalent to: irfft(cispi.(.-θ) .* z_div_conj_z.(rfft(zeropad(u, N))), N)
end

function incomplete_factorization_laurent_coeffs(u::AbstractVector{T}, ∇u::AbstractVector{T}, N::Int) where {T <: AbstractFloat}
    # Compute Laurent coefficients of the incomplete factorization:
    u, ∇u = zeropad(u, N), zeropad(∇u, N)
    p = get_plan_rfft(u)
    û, dû = p * u, p * ∇u
    @. û = dû / û
    return ldiv!(u, p, û) # output is equivalent to: irfft(rfft(zeropad(∇u, N)) ./ rfft(zeropad(u, N)), N)
end

function eigs_hankel(c::AbstractVector{T}; nev = 1) where {T <: AbstractFloat}
    # Declaring variable types helps inference since `eigs` is not type-stable,
    # and `eigen` may not be type-stable for e.g. BigFloat
    local D::Vector{T}, V::Matrix{T}
    nc = length(c)
    H = hankel(c)

    if nc <= 32 || nev >= nc ÷ 2
        D, V = eigen(Hermitian(Matrix(H))) # small problem, or need large fraction of eigenspace; convert Hankel to dense Matrix
        I = partialsortperm(D, 1:nev; by = abs, rev = true)
        D, V = D[I], V[:, I]
    elseif T <: Union{Float32, Float64}
        D, V = eigs(HermitianWrapper(H); nev, which = :LM, v0 = ones(T, nc) ./ nc, tol = nc * eps(T))
        D, V = convert(Vector{T}, D), convert(Matrix{T}, V)
    else
        F, _ = partialschur(HermitianWrapper(H); nev, which = ArnoldiMethod.LM(), tol = nc * eps(T))
        D, V = real(diag(F.R)), real(F.Q)
    end

    return D, V
end

function eigs_hankel_block(a::AbstractVector{T}, m, n; atol = 50 * eps(T)) where {T <: AbstractFloat}
    # Each Hankel matrix corresponds to one diagonal m - n = const in the CF-table;
    # when a diagonal intersects a square block, the eigenvalues on the
    # intersection are all equal. k and l tell you how many entries on the
    # intersection appear before and after the eigenvalues under consideration.
    # u is the corresponding eigenvector
    M = length(a) - 1

    if m - n + 1 < -M
        c = [zeros(T, -(m - n + 1) - M); a[(M+1).-abs.(-M:M)]]
    else
        c = a[(M+1).-abs.(m-n+1:M)]
    end

    nev = min(n + 10, length(c))
    D, V = eigs_hankel(c; nev)

    λ = D[n+1]
    u = V[:, n+1]

    k = 0
    while k < n && abs(abs(D[n-k]) - abs(λ)) < atol
        k += 1
    end

    l = 0
    while n + l + 2 < nev && abs(abs(D[n+l+2]) - abs(λ)) < atol
        l += 1
    end

    # Flag indicating if the function is actually rational
    is_rat = n + l + 2 == nev

    return λ, u, k, l, is_rat
end

function postprocess(p::AbstractVector{T}, λ::T, o::CFOptions{T}) where {T <: AbstractFloat}
    # Chop trailing zero coefficients to match parity
    p = paritychop(p, o.parity)

    # Clean up even/odd symmetric coefficients
    if o.parity === :even
        @views p[2:2:end] .= zero(T)
    elseif o.parity === :odd
        @views p[1:2:end] .= zero(T)
    end

    # Rescale the outputs
    p .*= o.vscale
    λ *= o.vscale

    return PolynomialApproximant(p, o.dom, λ)
end

function postprocess(p::AbstractVector{T}, q::AbstractVector{T}, λ::T, o::CFOptions{T}) where {T <: AbstractFloat}
    # Chop trailing zero coefficients to match parity
    if o.parity === :even
        p, q = paritychop(p, :even), paritychop(q, :even)
    elseif o.parity === :odd
        p, q = paritychop(p, :odd), paritychop(q, :even)
    end

    # Clean up even/odd symmetric coefficients
    if o.parity === :even
        @views p[2:2:end] .= zero(T) # even
        @views q[2:2:end] .= zero(T) # even
    elseif o.parity === :odd
        @views p[1:2:end] .= zero(T) # odd
        @views q[2:2:end] .= zero(T) # even
    end

    # Normalize the coefficients
    normalize_chebrational!(p, q)

    # Rescale the outputs
    p .*= o.vscale
    λ *= o.vscale

    return RationalApproximant(p, q, o.dom, λ)
end

#### Pade

function pade(c::AbstractVector{T}, m, n) where {T <: AbstractFloat}
    # Get the Chebyshev coefficients and pad if necessary
    l = max(m, n)
    a = zeropadresize(c, m + n + 1) #Note: Chebfun implementation notes that using random values instead of zero padding may be more stable, but that doesn't seem worth the non-determinism
    a[1] *= 2

    # Set up and solve Hankel system for denominator Laurent-Pade coefficients
    if n > 0
        top = m - n + 1 >= 0 ? a[m-n+2:m+1] : @views [a[-(m - n):-1:2]; a[1:m+1]]
        bot = @views a[m+1:m+n]
        rhs = @views -a[m+2:m+n+1]
        β = [one(T); reverse!(hankel(top, bot) \ rhs)]
    else
        β = ones(T, 1)
    end

    # Use convolution to compute numerator Laurent-Pade coefficients
    a[1] /= 2
    α = @views conv(a[1:l+1], β, l + 1) # Note: direct linear convolution is faster than FFT, since l = max(m, n) will never be very large in practice

    # Compute numerator Chebyshev-Pade coefficients
    p = zeros(T, m + 1)
    for j in 1:n+1, i in max(1, j - m):min(l + 1, j + m)
        p[abs(i - j)+1] += α[i] * β[j] # p[k] is the sum of the k'th and -k'th diagonals of D = α⋅βᵀ
    end

    # Compute denominator Chebyshev-Pade coefficients
    q = zeros(T, n + 1)
    q[1] = sum(abs2, β)
    for k in 2:n+1
        q[k] = @views 2 * dot(β[1:n+2-k], β[k:n+1])
    end

    # Normalize the coefficients
    normalize_chebrational!(p, q)

    return p, q
end

#### Minimax

Base.@kwdef struct MinimaxOptions{T <: AbstractFloat}
    "Domain of f(x)"
    dom::NTuple{2, T} = (-one(T), one(T))
    "Even/odd parity of f(x)"
    parity::Symbol = :generic
    "Upper bound on the maximum value of |f(x)|"
    vscale::T = one(T)
    "Maximum Chebyshev series degree. Series is truncated to this degree if necessary"
    maxdegree::Int = 2^10
    "Relative tolerance for detecting convergence"
    rtol::T = √eps(T)
    "Absolute tolerance for detecting convergence"
    atol::T = 5 * eps(T)
    "Maximum number of iterations"
    maxiter::Int = 25
    "Maximum number of Newton steps per iteration"
    maxsteps::Int = 10
    "Step size for Newton iteration"
    stepsize::T = one(T)
    "Clip step size of Newton iteration"
    stepclip::T = T(0.05)
    "Suppress warnings"
    quiet::Bool = false
end
const MinimaxOptionsFieldnames = Symbol[fieldnames(MinimaxOptions)...]

MinimaxOptions(c::AbstractVector{T}, dom::Tuple; kwargs...) where {T <: AbstractFloat} = MinimaxOptions{T}(; dom = check_endpoints(T.(dom)), parity = parity(c), vscale = vscale(c), kwargs...)
MinimaxOptions(o::MinimaxOptions{T}, c::AbstractVector{T}) where {T <: AbstractFloat} = (@set! o.parity = parity(c); @set! o.vscale = vscale(c); return o)

function minimax(fun::Fun, m::Int, n::Int; kwargs...)
    cf_kwargs, mm_kwargs = splitkwargs(kwargs, CFOptionsFieldnames, MinimaxOptionsFieldnames)
    p₀, q₀, _, _ = rationalcf(fun, m, n; cf_kwargs...)
    return minimax(fun, p₀, q₀; mm_kwargs...)
end
minimax(f, m::Int, n::Int; kwargs...) = minimax(ChebFun(f), m, n; kwargs...)
minimax(f, dom::Tuple, m::Int, n::Int; kwargs...) = minimax(ChebFun(f, dom), m, n; kwargs...)

# Provide initial guesses `p` and `q` directly
minimax(f, pq::AbstractVector...; kwargs...) = minimax(ChebFun(f), pq...; kwargs...)
minimax(f, dom::Tuple, pq::AbstractVector...; kwargs...) = minimax(ChebFun(f, dom), pq...; kwargs...)
minimax(fun::Fun, pq::AbstractVector...; kwargs...) = minimax(coefficients_and_endpoints(fun)..., pq...; kwargs...)
minimax(f::AbstractVector{T}, pq::AbstractVector{T}...; kwargs...) where {T <: AbstractFloat} = minimax(f, (-one(T), one(T)), pq...; kwargs...)
minimax(f::AbstractVector{T}, dom::Tuple, pq::AbstractVector{T}...; kwargs...) where {T <: AbstractFloat} = minimax(f, pq..., MinimaxOptions(f, dom; kwargs...))

struct PolynomialMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    δf::Vector{T}
    p::Vector{T}
    Eₓ::Vector{T}
    δp_δε::Vector{T}
    J::Matrix{T}
    rhs::Vector{T}
end
function PolynomialMinimaxWorkspace(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    m = length(p) - 1
    nx = m + 2 + (o.parity === :even || o.parity === :odd) # extra root expected for even/odd symmetry
    f, p = copy(f), copy(p)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    Eₓ, δp_δε = zeros(T, nx), zeros(T, nx)
    J, rhs = zeros(T, nx, m + 2), zeros(T, nx)
    return PolynomialMinimaxWorkspace(f, copy(f), p, Eₓ, δp_δε, J, rhs)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    (length(p) <= 1) && return minimax_constant_polynomial(f, o)
    m = length(p) - 1
    workspace = PolynomialMinimaxWorkspace(f, p, o)

    iter = 0
    @views while true
        local (; f, δf, p, Eₓ, δp_δε, J, rhs) = workspace
        δF = Fun(ChebSpace(T), δf)
        nx = length(rhs)

        # Update difference functional
        @. δf[1:m+1] = f[1:m+1] - p
        iszero(δF) && return postprocess(p, zero(T), o)

        # Compute new local extrema
        x, δFₓ, Sₓ = minimax_nodes(δF, δF', nx; o.quiet)
        nx₀ = length(x)
        nx′ = min(nx₀, nx)
        @. Eₓ[1:nx′] = abs(δFₓ[1:nx′]) # error at each root
        εmin, εmax = extrema(Eₓ[1:nx′])
        ε, σε = (εmax + εmin) / 2, (εmax - εmin) / 2

        if nx₀ != nx
            if iter == 0
                (εmax <= 50 * eps(T) * max(vscale(f), one(T))) && return postprocess(p, εmax, o) # initial approximant is sufficiently accurate
                !o.quiet && @warn "Initial polynomial approximant is not good enough to initialize minimax fine-tuning"
            end
            !o.quiet && @warn "Found $(nx₀) local extrema for degree $(m) $(o.parity) polynomial approximant, but expected $(nx)"
            return postprocess(p, εmax, o)
        end

        if σε <= max(o.rtol * ε, o.atol)
            # Minimax variance sufficiently small
            return postprocess(p, ε, o)
        end

        # Newton iteration on the (linear) residual equation
        #   G(p, ε) = P(x) + ε * S(x) - F(x) = 0
        residual_jacobian!(J, x, Sₓ, m) # Jacobian of residual equation: dG/d[p; ε]
        @. rhs = δFₓ - ε * Sₓ # right hand side: -G(p, ε)
        ldiv!(δp_δε, qr!(J), rhs) # solve for Newton step: [δp; δε] = J \ rhs

        # Damped Newton step
        δp = δp_δε[1:m+1]
        δε = δp_δε[m+2]
        @. p += o.stepsize * δp

        if abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol)
            # Newton step is small enough, so we're done
            return postprocess(p, ε, o)
        end

        if (iter += 1) > o.maxiter
            !o.quiet && @warn "Maximum number of Newton iterations reached"
            return postprocess(p, ε, o)
        end
    end
end

struct RationalMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    p::Vector{T}
    q::Vector{T}
    p₀::Vector{T}
    q₀::Vector{T}
    Pₓ::Vector{T}
    Qₓ::Vector{T}
    Eₓ::Vector{T}
    δp_δq_δε::Vector{T}
    J::Matrix{T}
    rhs::Vector{T}
end
function RationalMinimaxWorkspace(f::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    m, n = length(p) - 1, length(q) - 1
    nx = m + n + 2 + (o.parity === :even || o.parity === :odd) # extra root expected for even/odd symmetry
    f, (p, q) = copy(f), normalize_unitrational(p, q)
    p₀, q₀ = copy(p), copy(q)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    Pₓ, Qₓ, Eₓ, δp_δq_δε = ntuple(_ -> zeros(T, nx), 4)
    J, rhs = zeros(T, nx, m + n + 2), zeros(T, nx) # J may have more rows than columns for symmetric inputs
    return RationalMinimaxWorkspace(f, p, q, p₀, q₀, Pₓ, Qₓ, Eₓ, δp_δq_δε, J, rhs)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    (length(q) <= 1) && return minimax(f, p, o)
    (length(p) <= 1 && o.parity === :odd) && return minimax_constant_polynomial(f, o)
    m, n = length(p) - 1, length(q) - 1
    workspace = RationalMinimaxWorkspace(f, p, q, o)

    iter = 0
    @views while true
        local (; f, p, q, p₀, q₀, Pₓ, Qₓ, Eₓ, δp_δq_δε, J, rhs) = workspace
        F = Fun(ChebSpace(T), f)
        P = Fun(ChebSpace(T), p)
        Q = Fun(ChebSpace(T), q)
        δF = x -> F(x) - P(x) / Q(x)
        nx = length(rhs)

        # Compute new extremal nodes
        ∇δF_pseudo = (F' * Q - P') * Q + P * Q' # numerator of (F - P/Q)' = F' - (P'Q - PQ')/Q²; note that we assume Q(x) has no roots in [-1, 1]
        x, δFₓ, Sₓ = minimax_nodes(δF, ∇δF_pseudo, nx; o.quiet)
        nx₀ = length(x)
        nx′ = min(nx₀, nx)

        @. Pₓ[1:nx′] = P(x[1:nx′])
        @. Qₓ[1:nx′] = Q(x[1:nx′])
        @. Eₓ[1:nx′] = abs(δFₓ[1:nx′]) # error at each root
        εmin, εmax = extrema(Eₓ[1:nx′])
        ε, σε = (εmax + εmin) / 2, (εmax - εmin) / 2

        if nx₀ != nx
            if iter == 0
                (εmax <= 50 * eps(T) * max(vscale(f), one(T))) && return postprocess(p, q, εmax, o) # approximant is sufficiently accurate
                !o.quiet && @warn "Initial rational approximant is not good enough to initialize minimax fine-tuning"
            end
            !o.quiet && @warn "Found $(nx₀) local extrema for type $((m, n)) $(o.parity) rational approximant, but expected $(nx)"
            return postprocess(p, q, εmax, o)
        end

        if σε <= max(o.rtol * ε, o.atol)
            # Minimax variance sufficiently small
            return postprocess(p, q, ε, o)
        end

        # Newton iteration on the residual equation
        #   G(p, q, ε) = P(x) + (ε * S(x) - F(x)) * Q(x) = 0
        for i in 1:o.maxsteps
            residual_jacobian!(J, x, Pₓ, Qₓ, δFₓ, Sₓ, ε, m, n) # Jacobian of residual equation: dG/d[p; q; ε]
            @. rhs = (δFₓ - ε * Sₓ) * Qₓ # right hand side: -G(p, q, ε)
            ldiv!(δp_δq_δε, qr!(J), rhs) # solve for Newton step: [δp; δq; δε] = J \ rhs

            # Damped Newton step
            δp = δp_δq_δε[1:m+1]
            δq = δp_δq_δε[m+2:m+n+1]
            δε = δp_δq_δε[m+n+2]

            copy!.((p₀, q₀), (p, q))
            @. p += clamp(o.stepsize * δp, -o.stepclip, o.stepclip)
            @. q[2:n+1] += clamp(o.stepsize * δq, -o.stepclip, o.stepclip)
            ε += o.stepsize * δε

            if isempty(chebroots(Q; no_pts = max(12, n + 1)))
                # No poles in the denominator of the rational approximant
                normalize_unitrational!(p, q)
                @. Pₓ = P(x)
                @. Qₓ = Q(x)
                @. δFₓ = δF(x)
                @. Eₓ = abs(δFₓ) # error at each root

                εmin, εmax = extrema(Eₓ)
                σε = (εmax - εmin) / 2
                res = maximum(abs, @. δFₓ - ε * Sₓ) # maximum minimax residual

                if res <= max(o.rtol * ε, o.atol) || σε <= max(o.rtol * ε, o.atol)
                    # Minimax residual is small enough, so we're done
                    break
                elseif abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol) || maximum(abs, δq) <= max(o.rtol * maximum(abs, q), o.atol)
                    # Newton step is small enough, so we're done
                    break
                end
            else
                !o.quiet && @warn "Newton step created pole(s) in the denominator of the rational approximant; exiting early"
                copy!.((p, q), (p₀, q₀))
                return postprocess(p, q, ε, o)
            end
        end

        if (iter += 1) > o.maxiter
            !o.quiet && @warn "Maximum number of Newton iterations reached"
            return postprocess(p, q, ε, o)
        end
    end
end

@views function residual_jacobian!(J::AbstractMatrix{T}, x::AbstractVector{T}, Sₓ::AbstractVector{T}, m::Int) where {T <: AbstractFloat}
    # Compute Jacobian of residual equation:
    #   G(p, ε) = P(x) + ε * S(x) - F(x) = 0
    nx = length(x)
    @assert nx >= m + 2
    @assert size(J) == (nx, m + 2)
    @assert length(Sₓ) == nx

    # First m+1 columns: dG/dpⱼ = Tⱼ(x), j = 0, ..., m
    @. J[:, 1] = one(T) # T₀(x) = 1
    if m >= 1
        @. J[:, 2] = x # T₁(x) = x
    end
    for j in 3:m+1
        @. J[:, j] = 2x * J[:, j-1] - J[:, j-2] # Tⱼ(x) = 2x * Tⱼ₋₁(x) - Tⱼ₋₂(x)
    end

    # Last column: dG/dε = S(x)
    @. J[:, m+2] = Sₓ

    return J
end

@views function residual_jacobian!(J::AbstractMatrix{T}, x::AbstractVector{T}, Pₓ::AbstractVector{T}, Qₓ::AbstractVector{T}, δFₓ::AbstractVector{T}, Sₓ::AbstractVector{T}, ε::T, m::Int, n::Int) where {T <: AbstractFloat}
    # Compute Jacobian of residual equation:
    #   G(p, q, ε) = P(x) + (ε * S(x) - F(x)) * Q(x) = 0
    nx = length(x)
    @assert nx >= m + n + 2
    @assert size(J) == (nx, m + n + 2)
    @assert length(Pₓ) == length(Qₓ) == length(δFₓ) == length(Sₓ) == nx

    # First m+1 columns: dG/dpⱼ = Tⱼ(x), j = 0, ..., m
    @. J[:, 1] = one(T) # T₀(x) = 1
    if m >= 1
        @. J[:, 2] = x # T₁(x) = x
    end
    for j in 3:m+1
        @. J[:, j] = 2x * J[:, j-1] - J[:, j-2] # Tⱼ(x) = 2x * Tⱼ₋₁(x) - Tⱼ₋₂(x)
    end

    # Next n columns: dG/dqⱼ = (ε * S(x) - F(x)) * Tⱼ(x), j = 1, ..., n
    @. J[:, m+2] = x # T₁(x) = x
    if n >= 2
        @. J[:, m+3] = 2x^2 - 1 # T₂(x) = 2x² - 1
    end
    for j in m+4:m+n+1
        @. J[:, j] = 2x * J[:, j-1] - J[:, j-2] # Tⱼ(x) = 2x * Tⱼ₋₁(x) - Tⱼ₋₂(x)
    end
    @. J[:, m+2:m+n+1] *= (ε * Sₓ - δFₓ) - Pₓ / Qₓ

    # Last column: dG/dε = S(x) * Q(x)
    @. J[:, m+n+2] = Sₓ * Qₓ

    return J
end

function minimax_nodes(f, ∇f_pseudo::Fun{<:Chebyshev, T}, nnodes::Int; atol::T = 100 * eps(T), quiet::Bool = false) where {T <: AbstractFloat}
    @assert nnodes >= 2

    # Local extremal nodes. Note: Sₓ = +1 for local maxima, -1 for local minima, but it is *not* necessarily the sign of f(x)
    x, Sₓ = local_extremal_nodes(∇f_pseudo; no_pts = max(12, nnodes + 1))
    fₓ = @. f(x)
    check_signs(Sₓ; quiet)

    # Check endpoints for extrema
    x₀, x₁ = endpoints(domain(∇f_pseudo))
    f₀, f₁ = f(x₀), f(x₁)
    isempty(x) && return [x₀; x₁], [f₀; f₁], f₀ <= f₁ ? T[-1, 1] : T[1, -1] # no interior extrema

    # Add both endpoints, if they are not already present
    if abs(x[1] - x₀) > atol
        x = pushfirst!(x, x₀)
        fₓ = pushfirst!(fₓ, f₀)
        Sₓ = pushfirst!(Sₓ, -Sₓ[1])
    end
    if abs(x[end] - x₁) > atol
        x = push!(x, x₁)
        fₓ = push!(fₓ, f₁)
        Sₓ = push!(Sₓ, -Sₓ[end])
    end
    length(x) <= nnodes && return x, fₓ, Sₓ # out of points to add, so we're done

    # Have too many points; choose the streak of `nnodes` nodes with the maximum signed sum of errors
    ibest, errbest = 0, T(-Inf)
    for i in 1:length(x)-nnodes+1
        err = sum(@views Sₓ[i:i+nnodes-1] .* fₓ[i:i+nnodes-1]) # signed sum of errors rewards large errors with the correct sign
        if err > errbest
            ibest, errbest = i, err
        end
    end
    x = x[ibest:ibest+nnodes-1]
    fₓ = fₓ[ibest:ibest+nnodes-1]
    Sₓ = Sₓ[ibest:ibest+nnodes-1]

    return x, fₓ, Sₓ
end

function local_extremal_nodes(∇f_pseudo::Fun{<:Chebyshev, T}; kwargs...) where {T <: AbstractFloat}
    # Compute local extrema of a function f(x).
    #   - ∇f_pseudo(x) need only be pointwise proportional to ∇f(x), i.e. ∇f_pseudo(x) = ∇f(x) * g(x), where g(x) != 0 ∀ x ∈ domain(f)
    #   - ∇²f_pseudo(x) = d/dx ∇f_pseudo(x), and Sₓ = +1 for local maxima, -1 for local minima
    ∇f = coefficients(∇f_pseudo)
    vscale(∇f) == zero(T) && return T[], T[] # zero function
    ∇²f_pseudo = ∇f_pseudo'

    # Use at least 64-bit precision for initial node computation
    ∇f64 = lazychopcoeffs(convert(Vector{Float64}, ∇f); rtol = eps(Float64))
    ∇f64_pseudo = Fun(ChebSpace(Float64), ∇f64)
    x = convert(Vector{T}, chebroots(∇f64_pseudo; kwargs...))

    # Compute sign of local extrema; +1 for local maxima, -1 for local minima
    Sₓ = @. -floatsign(∇²f_pseudo(x))

    if precision(T) > precision(Float64)
        if check_signs(Sₓ; quiet = true)
            # Found alternating sequence of extrema; refine the initial nodes with higher precision
            x, Sₓ = refine_roots!(∇f_pseudo, ∇²f_pseudo, x)
        else
            # Computing nodes in lower precision failed to produce an alternating sequence of extrema; try again in higher precision
            x = chebroots(∇f; kwargs...)
            Sₓ = @. -floatsign(∇²f_pseudo(x))
        end
    end
    isempty(x) && return T[], T[] # no extrema found

    # Cleanup: ensure roots are unique, sorted, and contained in [-1, 1]
    @. x = clamp(x, -one(T), one(T))
    if !issorted(x)
        I = sortperm(x)
        x, Sₓ = x[I], Sₓ[I]
    end
    i = 1
    while i < length(x)
        if x[i] == x[i+1]
            deleteat!(x, i + 1)
            deleteat!(Sₓ, i + 1)
        else
            i += 1
        end
    end

    return x, Sₓ
end

function refine_roots!(∇fun::Fun, ∇²fun::Fun, x::AbstractVector{T}) where {T <: AbstractFloat}
    # Refine roots using Newton's method
    isempty(x) && return T[], T[]
    Δx, ∇fₓ, ∇²fₓ, Sₓ = zero(x), zero(x), zero(x), zero(x)
    Δx_max = T(Inf)
    maxiter = ceil(Int, log2(-log2(eps(T)))) # relative error should be ~eps(Float64) to start, and Newton's method squares the error each iteration
    for i in 1:maxiter
        ∇fₓ .= extrapolate.((∇fun,), x)
        ∇²fₓ .= extrapolate.((∇²fun,), x)
        @. Sₓ = -floatsign(∇²fₓ) # +1 for local maximum, -1 for local minimum
        @. Δx = ∇fₓ / ∇²fₓ # note: Δx is invariant to the scale of ∇fun
        @. x -= Δx
        Δx_max, Δx_max_last = maximum(abs, Δx), Δx_max
        (Δx_max <= 5 * eps(T) || (Δx_max < √eps(T) && Δx_max > Δx_max_last / 2)) && break
    end
    return x, Sₓ
end

function minimax_constant_polynomial(f::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    lo, hi = chebrange(f)
    p, ε = [T(lo + hi) / 2], T(hi - lo) / 2
    return postprocess(p, ε, o)
end

postprocess(p::AbstractVector{T}, ε::T, o::MinimaxOptions{T}) where {T <: AbstractFloat} = PolynomialApproximant(p, o.dom, ε)
postprocess(p::AbstractVector{T}, q::AbstractVector{T}, ε::T, o::MinimaxOptions{T}) where {T <: AbstractFloat} = RationalApproximant(normalize_chebrational!(p, q)..., o.dom, ε)

function check_signs(S; quiet = false)
    pass = all(S[i] == -S[i+1] for i in 1:length(S)-1)
    !quiet && !pass && @warn "Newton iterations converged, but error signs are not alternating"
    return pass
end

#### Linear algebra utilities

hankel(c::AbstractVector) = Hankel(zeropad(c, 2 * length(c) - 1))
hankel(c::AbstractVector, r::AbstractVector) = Hankel(c, r)
toeplitz(c::AbstractVector) = Toeplitz(c, c)

@inline function z_div_conj_z(z::Complex)
    # Efficiently, safely, and accurately compute z / conj(z). Returns 1 if z == 0.
    a, b = reim(z)
    r² = a^2 + b^2
    w = Complex((a - b) * (a + b), 2 * a * b) / r²
    return ifelse(r² == 0, one(w), w)
end

function conv(a::AbstractVector, b::AbstractVector, L::Int = length(a) + length(b) - 1)
    N, M = length(a), length(b)
    @assert 0 <= L <= N + M - 1 "Invalid output length"
    T = typeof(one(eltype(a)) * one(eltype(b)))
    c = zeros(T, L)
    for i in 1:N, j in 1:min(M, L - i + 1)
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

# Cache rFFT plans. Worthwhile since we are always using the same power-of-2 sized rFFTs
const PLAN_RFFT_CACHE = Dict{Tuple{DataType, Int}, Plan}()
get_plan_rfft(x::AbstractVector{T}) where {T <: AbstractFloat} = get!(() -> plan_rfft(x), PLAN_RFFT_CACHE, (T, length(x)))::Plan{T}

# Dummy wrapper to ensure we don't hit generic fallbacks
struct HermitianWrapper{T <: AbstractFloat, A <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::A
end
LinearAlgebra.size(H::HermitianWrapper) = size(H.A)
LinearAlgebra.eltype(H::HermitianWrapper) = eltype(H.A)
LinearAlgebra.parent(H::HermitianWrapper) = H.A
LinearAlgebra.issymmetric(H::HermitianWrapper) = true
LinearAlgebra.ishermitian(H::HermitianWrapper) = true
LinearAlgebra.transpose(H::HermitianWrapper) = H
LinearAlgebra.adjoint(H::HermitianWrapper) = H
LinearAlgebra.mul!(y::AbstractVector, H::HermitianWrapper, x::AbstractVector, α::Number, β::Number) = mul!(y, H.A, x, α, β)

#### Helper functions

ChebSpace(::Type{T}) where {T <: AbstractFloat} = Chebyshev(ChebyshevInterval{T}())
ChebSpace(::Type{T}, dom::Tuple) where {T <: AbstractFloat} = Chebyshev(ClosedInterval{T}(dom...))

ChebFun(f) = ChebFun(Float64, f)
ChebFun(f::AbstractVector) = ChebFun(float(eltype(f)), f)
ChebFun(f, dom::Tuple) = ChebFun(promote_type(map(typeof ∘ float, dom)...), f, dom)
ChebFun(f::AbstractVector, dom::Tuple) = ChebFun(float(eltype(f)), f, dom)
ChebFun(::Type{T}, f) where {T <: AbstractFloat} = ChebFun(T, f, (-one(T), one(T)))
ChebFun(::Type{T}, f, dom::Tuple) where {T <: AbstractFloat} = Fun(f, ChebSpace(T, dom))
ChebFun(::Type{T}, f::AbstractVector, dom::Tuple) where {T <: AbstractFloat} = Fun(ChebSpace(T, dom), convert(Vector{T}, f))

function check_endpoints(dom::Tuple)
    @assert length(dom) == 2 "Domain must be a tuple of length 2"
    @assert dom[1] < dom[2] "Domain must be increasing"
    @assert all(isfinite, dom) "Domain must be finite"
    return promote(map(float, dom)...)
end
check_endpoints(dom::Union{Interval, ChebyshevInterval}) = check_endpoints(endpoints(dom))
check_endpoints(dom::Chebyshev) = check_endpoints(domain(dom))
check_endpoints(fun::Fun) = check_endpoints(domain(fun))

function check_fun(fun::Fun)
    @assert isreal(fun) "Only real-valued functions are supported"
    @assert space(fun) isa Chebyshev "Only Chebyshev spaces are supported"
    check_endpoints(fun)
    return fun
end

function coefficients_and_endpoints(fun::Fun)
    c = coefficients(check_fun(fun))
    dom = check_endpoints(fun)
    return c, dom
end

function unzip(c::AbstractVector; even::Bool)
    @assert length(c) >= 1
    a = zeros(eltype(c), 2 * length(c) - even)
    @views a[1+!even:2:end] .= c
    return a
end

function zeropad!(x::AbstractVector, n::Int)
    (nx = length(x)) >= n && return x
    resize!(x, n)[nx+1:end] .= zero(eltype(x))
    return x
end
zeropad(x::AbstractVector, n::Int) = zeropad!(x[1:min(end, n)], n)

zeropadresize!(x::AbstractVector, n::Int) = resize!(zeropad!(x, n), n)
zeropadresize(x::AbstractVector, n::Int) = zeropadresize!(x[1:min(end, n)], n)

function splitkwargs(kwargs, set1, set2)
    kws1 = intersect(keys(kwargs), set1) # keyword arguments in set1
    kws2 = setdiff(keys(kwargs), setdiff(set1, set2)) # filter out keyword arguments exclusive to set2
    return kwargs[kws1], kwargs[kws2]
end

@inline floatsign(x::T) where {T <: AbstractFloat} = ifelse(x < zero(T), -one(T), one(T))
@inline floattype(::Type{T}) where {T <: Number} = float(real(T))
@inline floattype(fun::Fun) = floattype(eltype(coefficients(fun)))

evalrat(x, p, q) = evalpoly(x, p) / evalpoly(x, q)
evalrat(x, p) = evalpoly(x, p)

#### Polynomial utilities

function poly(c::AbstractVector{T}, dom::Tuple = (-1, 1); transplant = dom != (-1, 1)) where {T}
    # Convert a vector of Chebyshev coefficients to a vector of monomial coefficients,
    # optionally linearly transplanted to a new domain.
    @assert length(dom) == 2 "Domain must be a tuple of length 2"
    Q = ChebyshevT{T, :x}(c) # polynomial in Chebyshev basis on [-1, 1]
    P = Polynomial{T, :x}(Q) # polynomial in monomial basis on [-1, 1]
    if transplant && dom != (-1, 1)
        a, b = T.(dom)
        t = Polynomial{T, :x}([-(a + b) / (b - a), 2 / (b - a)]) # polynomial mapping [a, b] -> [-1, 1]
        P = P(t) # polynomial in monomial basis on [a, b]
    end
    return Polynomials.coeffs(P)
end
poly(fun::Fun; kwargs...) = poly(coefficients_and_endpoints(fun)...; kwargs...)

polyder(c::AbstractVector) = polyder(Polynomial(c))
polyder(p::Polynomial) = Polynomials.coeffs(Polynomials.derivative(p))

polyroots(c::AbstractVector) = polyroots(Polynomial(c))
polyroots(p::Polynomial) = complex(Polynomials.roots(p))

function scale_rational!(p::AbstractVector{T}, q::AbstractVector{T}, q₀::T) where {T <: AbstractFloat}
    p ./= q₀
    q ./= q₀
    return p, q
end

normalize_rational!(p::AbstractVector, q::AbstractVector) = scale_rational!(p, q, first(q))
normalize_rational(p::AbstractVector, q::AbstractVector) = normalize_rational!(copy(p), copy(q))

normalize_chebpoly!(p::AbstractVector) = p ./= chebeval_at_midpoint(p)
normalize_chebpoly(p::AbstractVector) = normalize_chebpoly!(copy(p))

normalize_chebrational!(p::AbstractVector, q::AbstractVector) = scale_rational!(p, q, chebeval_at_midpoint(q))
normalize_chebrational(p::AbstractVector, q::AbstractVector) = normalize_chebrational!(copy(p), copy(q))

normalize_unitrational!(p::AbstractVector, q::AbstractVector) = scale_rational!(p, q, √(sum(abs2, p) + sum(abs2, q)))
normalize_unitrational(p::AbstractVector, q::AbstractVector) = normalize_unitrational!(copy(p), copy(q))

function parity(c::AbstractVector{T}; rtol = eps(T)) where {T <: AbstractFloat}
    (length(c) <= 1) && return :even # constant function
    scale = vscale(c)
    (scale == zero(T)) && return :even
    oddscale = @views vscale(c[1:2:end])
    (oddscale <= rtol * scale) && return :odd
    evenscale = @views vscale(c[2:2:end])
    (evenscale <= rtol * scale) && return :even
    return :generic
end
parity(fun::Fun; kwargs...) = parity(coefficients(fun); kwargs...)

# Chop small coefficients from the end of a Chebyshev series
function lazychopcoeffs(c::AbstractVector{T}; rtol::T = eps(T), atol::T = zero(T), parity::Symbol = :generic) where {T <: AbstractFloat}
    scale = vscale(c)
    l = length(c)
    while l > 1 && abs(c[l]) <= max(rtol * scale, atol)
        l -= 1
    end
    if parity === :even || parity === :odd
        return @views lazyparitychop(c[1:l], parity)
    else
        return @views c[1:l]
    end
end
chopcoeffs(c::AbstractVector; kwargs...) = convert(Vector, lazychopcoeffs(c; kwargs...))
chopcoeffs(fun::Fun; kwargs...) = Fun(space(fun), chopcoeffs(coefficients(fun); kwargs...))

function lazyparitychop(c::AbstractVector, parity::Symbol)
    M = length(c) - 1
    if parity === :even
        isodd(M) && (M -= 1) # ensure M is even
    elseif parity === :odd
        M > 0 && iseven(M) && (M -= 1) # ensure M is odd, or zero
    end
    return @views c[1:M+1]
end
paritychop(c::AbstractVector, parity::Symbol) = convert(Vector, lazyparitychop(c, parity))

# Crude bound on infinity norm of `fun`
vscale(fun::Fun) = vscale(coefficients(fun))
vscale(c::AbstractVector{T}) where {T <: AbstractFloat} = sum(abs, c; init = zero(T))

function chebeval_at_midpoint(c::AbstractVector{T}) where {T <: AbstractFloat}
    # Compute f(0) for a Chebyshev series f(x) = ∑ₖ cₖ Tₖ(x).
    f₀ = zero(T)
    c′ = @views c[1:2:end] # even coefficients
    for k in length(c′)-1:-1:0
        c₂ₖ = c′[k+1]
        f₀ += ifelse(iseven(k), c₂ₖ, -c₂ₖ) # Tₖ(0) = {+1 if k ≡ 0 (mod 4), -1 if k ≡ 2 (mod 4)}
    end
    return f₀
end
chebeval_at_midpoint(f::Fun) = chebeval_at_midpoint(coefficients(f))

function chebeval_at_endpoints(c::AbstractVector{T}) where {T <: AbstractFloat}
    # Compute f(-1) and f(+1) for a Chebyshev series f(x) = ∑ₖ cₖ Tₖ(x).
    f⁻ = f⁺ = zero(T)
    for k in length(c)-1:-1:0
        cₖ = c[k+1]
        f⁻ += ifelse(iseven(k), cₖ, -cₖ) # Tₖ(-1) = {+1 if k ≡ 0 (mod 2), -1 if k ≡ 2 (mod 2)}
        f⁺ += cₖ # Tₖ(+1) = +1
    end
    return f⁻, f⁺
end
chebeval_at_endpoints(f::Fun) = chebeval_at_endpoints(coefficients(f))

function chebrange(f::AbstractVector{T}) where {T <: AbstractFloat}
    # Compute the infinity norm of a Fun
    fun = Fun(ChebSpace(T), f)
    fa, fb = minmax(chebeval_at_endpoints(fun)...)
    x = chebroots(fun')
    isempty(x) && return fa, fb
    flo, fhi = extrema(fun, x)
    return min(fa, flo), max(fb, fhi)
end
chebrange(fun::Fun) = chebrange(coefficients(fun))
chebinfnorm(fun::Fun) = max(abs.(chebrange(fun))...)

#### Rootfinding utilities

function chebroots(c₀::AbstractVector{T}; kwargs...) where {T <: AbstractFloat}
    # Using bisection on the interval [-1, 1] directly tends to be much faster than computing
    # all of the eigenvalues of the colleague matrix, since in general a degree `d` degree polynomial
    # has `d` complex roots but only `log(d)` real roots.
    scale = vscale(c₀)
    scale == zero(T) && return T[] # zero function
    c = lazychopcoeffs(c₀ ./ scale; rtol = zero(T), atol = zero(T)) # normalize coefficients and prune exact zeros; we want scale-independent roots
    return Roots.find_zeros(Fun(ChebSpace(T), c), (-one(T), one(T)); kwargs...)
end
chebroots(fun::Fun; kwargs...) = chebroots(coefficients(fun); kwargs...)

# The below code is largely taken from ApproxFunOrthogonalPolynomials.jl, adjusted to work for generic types:
#   https://github.com/JuliaApproximation/ApproxFunOrthogonalPolynomials.jl/blob/90d8cbe03adb2d95c50c165377d4f20a02d2d1e6/src/roots.jl#L82
# TODO: Extend this code to work with complex coefficients and upstream to ApproxFunOrthogonalPolynomials.jl

function colleague_chebroots(c::AbstractVector{T}; atol = 100 * eps(T)) where {T <: AbstractFloat}
    scale = vscale(c)
    scale == zero(T) && return T[] # zero function
    c = chopcoeffs(c; rtol = zero(T), atol = zero(T)) # prune exact zeros
    c ./= scale

    # Recursively compute roots on subintervals
    r = recurse_colleague_chebroots(c, atol)

    # Check endpoints, which are prone to inaccuracy
    f⁻, f⁺ = chebeval_at_endpoints(c)
    (isempty(r) || !isapprox(last(r), one(T); atol = atol)) && (abs(f⁺) < atol) && push!(r, one(T))
    (isempty(r) || !isapprox(first(r), -one(T); atol = atol)) && (abs(f⁻) < atol) && pushfirst!(r, -one(T))

    return r
end
colleague_chebroots(f::Fun; kwargs...) = colleague_chebroots(coefficients(f); kwargs...)

function recurse_colleague_chebroots(c₀::AbstractVector{T}, atol::T) where {T <: AbstractFloat}
    # Compute the real roots contained in [-1, 1] of a polynomial in the Chebyshev basis
    c = lazychopcoeffs(c₀; rtol = eps(T))
    n = length(c)

    if n == 0
        # Zero function
        return T[]

    elseif n == 1
        # Constant function
        return c[1] == zero(T) ? zeros(T, 1) : T[]

    elseif n == 2
        # Linear function
        r₀ = -c[1] / c[2]
        return abs(imag(r₀)) > atol || abs(real(r₀) > 1 + atol) ? T[] : T[clamp(real(r₀), -one(T), one(T))]

    elseif n <= 70
        # Polynomial degree is small enough; compute roots directly using the colleague matrix
        rc = eigvals(colleague_matrix(c)) # all complex roots of the polynomial
        rc = filter!(z -> abs(imag(z)) < atol && abs(real(z)) <= 1 + atol, rc) # keep roots that lie in [-1 - atol, 1 + atol] x [-atol, atol]
        return clamp.(real.(rc), -one(T), one(T)) # return real part clamped to [-1, 1]

    else
        # Recursively subdivide the interval [-1,1] into [-1, x₀], [x₀, 1]
        x₀ = T(-0.004849834917525)
        a⁻, b⁻ = (x₀ + 1) / 2, (x₀ - 1) / 2
        a⁺, b⁺ = (1 - x₀) / 2, (x₀ + 1) / 2
        f = Fun(ChebSpace(T), c)
        f1 = Fun(x -> f(muladd(a⁻, x, b⁻)), ChebSpace(T))
        f2 = Fun(x -> f(muladd(a⁺, x, b⁺)), ChebSpace(T))

        # Recurse and map roots back to original interval
        r1 = Threads.@spawn muladd.(a⁻, recurse_colleague_chebroots(coefficients(f1), 2 * atol)::Vector{T}, b⁻) # absolute tolerance on the stretched out half interval is double the absolute tolerance on the original interval
        r2 = muladd.(a⁺, recurse_colleague_chebroots(coefficients(f2), 2 * atol)::Vector{T}, b⁺)
        return [fetch(r1)::Vector{T}; r2]
    end
end

function colleague_matrix(c::AbstractVector{T}) where {T <: Number}
    # Form the colleague matrix from a vector `c` of Chebyshev coefficients
    n = length(c) - 1
    A = zeros(T, n, n)

    for k in 1:n-1
        A[k+1, k] = T(0.5)
        A[k, k+1] = T(0.5)
    end
    for k in 1:n
        A[1, end-k+1] -= T(0.5) * c[k] / c[end]
    end
    A[n, n-1] = one(T)

    return A
end

#### Precompile

function precompile()
    # Note: BigFloat is not worth precompiling; it makes precompilation take way longer,
    # and TTFX is not dominated by precompilation time for BigFloat anyway.
    for T in [Float32, Float64, Double64]
        # Oscillatory function with many zeros and relatively long Chebyshev series
        fun = ChebFun(T, x -> exp(-4x^2) * (1 + 2 * sinpi(10x)))
        @assert length(chebroots(fun)) == 20
    end
    for T in [Float32, Float64, Double64], f in [exp, cospi, sinpi], dom in [(-1.0f0, 1.0f0), (-0.97f0, 1.32f0)], (m, n) in [(2, 0), (3, 0), (2, 3), (3, 2)]
        # Minimax with generic, even, and odd functions with relatively short Chebyshev series.
        # Note: minimax calls rationalcf/polynomialcf internally, so don't need to separately precompile those.
        minimax(f, T.(dom), m, n)
    end
end

@compile_workload begin
    precompile()
end

end # module CaratheodoryFejerApprox
