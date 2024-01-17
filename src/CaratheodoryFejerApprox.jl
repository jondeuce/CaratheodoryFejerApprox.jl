#=
CaratheodoryFejerApprox.jl is based largely on the following papers:

    [1] Gutknecht MH, Trefethen LN. Real Polynomial Chebyshev Approximation by the Carathéodory–Fejér method. SIAM J Numer Anal 1982; 19: 358–371.
    [2] Trefethen LN, Gutknecht MH. The Carathéodory–Fejér Method for Real Rational Approximation. SIAM J Numer Anal 1983; 20: 420–436.
    [3] Henrici P. Fast Fourier Methods in Computational Complex Analysis. SIAM Rev 1979; 21: 481–527.

As well as the following files from Chebfun v5:

    chebfun/@chebfun/cf.m
    chebfun/@chebfun/chebpade.m

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

using ApproxFun: ApproxFun, Chebyshev, ChebyshevInterval, Fun, Interval, coefficients, domain, endpoints, ncoefficients, space
using ArnoldiMethod: ArnoldiMethod, partialschur
using Arpack: Arpack, eigs
using FFTW: FFTW, irfft, rfft
using GenericFFT: GenericFFT # defines generic methods for FFTs
using GenericLinearAlgebra: GenericLinearAlgebra # defines generic method for `LinearAlgebra.eigen` and `Polynomials.roots`
using LinearAlgebra: LinearAlgebra, Hermitian, cholesky!, cond, diag, dot, eigen, ldiv!, mul!, qr!
using Polynomials: Polynomials, ChebyshevT, Polynomial
using Setfield: Setfield, @set!
using Statistics: Statistics, mean, std
using ToeplitzMatrices: ToeplitzMatrices, Toeplitz, Hankel

export minimax, polynomialcf, rationalcf
export chebcoeffs, monocoeffs

struct RationalApproximant{T <: AbstractFloat}
    p::AbstractVector{T}
    q::AbstractVector{T}
    dom::NTuple{2, T}
    err::T
end
PolynomialApproximant(p::AbstractVector{T}, dom::NTuple{2, T}, err::T) where {T <: AbstractFloat} = RationalApproximant(p, ones(T, 1), dom, err)

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
    println(io, "  Type:   (m, n) = (", m, ", ", n, ")")
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

polynomialcf(f, m::Int; kwargs...) = polynomialcf(Fun(f), m; kwargs...)
polynomialcf(f, dom::Tuple, m::Int; kwargs...) = polynomialcf(Fun(f, Chebyshev(Interval(dom...))), m; kwargs...)
polynomialcf(fun::Fun, m::Int; kwargs...) = polynomialcf(coefficients_and_endpoints(fun)..., m; kwargs...)
polynomialcf(c::AbstractVector{T}, m::Int; kwargs...) where {T <: AbstractFloat} = polynomialcf(c, (-one(T), one(T)), m; kwargs...)
polynomialcf(c::AbstractVector{T}, dom::Tuple, m::Int; kwargs...) where {T <: AbstractFloat} = polynomialcf(c, m, CFOptions(c, dom; kwargs...))

rationalcf(f, m::Int, n::Int; kwargs...) = rationalcf(Fun(f), m, n; kwargs...)
rationalcf(f, dom::Tuple, m::Int, n::Int; kwargs...) = rationalcf(Fun(f, Chebyshev(Interval(dom...))), m, n; kwargs...)
rationalcf(fun::Fun, m::Int, n::Int; kwargs...) = rationalcf(coefficients_and_endpoints(fun)..., m, n; kwargs...)
rationalcf(c::AbstractVector{T}, m::Int, n::Int; kwargs...) where {T <: AbstractFloat} = rationalcf(c, (-one(T), one(T)), m, n; kwargs...)
rationalcf(c::AbstractVector{T}, dom::Tuple, m::Int, n::Int; kwargs...) where {T <: AbstractFloat} = rationalcf(c, m, n, CFOptions(c, dom; kwargs...))

function polynomialcf(c::AbstractVector{T}, m::Int, o::CFOptions) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested polynomial degree m = $(m) must be non-negative"

    iszero(o.vscale) && return postprocess(c, zeros(T, 1), zero(T), o) # trivial function
    c = @views c[1:min(end - 1, o.maxdegree)+1] ./ o.vscale # truncate to maxdegree and normalize scale
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
    (m == M) && return postprocess(c, c[1:M+1], zero(T), o)
    (m == M - 1) && return postprocess(c, c[1:M], abs(c[M+1]), o)
    (m == 0) && return postprocess(c, polynomialcf_constant(c)..., o)

    if o.parity === :even
        # If f(x) is even, we can reduce the problem to polynomial approximation of g(x) = f(T₂(x))
        c′ = @views c[1:2:end]
        p′, _, _, s = polynomialcf(c′, m ÷ 2, CFOptions(o, c′))
        p = unzip(p′; even = true)
        return postprocess(c, p, s, o)
    end

    cm = @views c[m+2:M+1]
    D, V = eigs_hankel(cm; nev = 1)
    s, u = only(D), vec(V)

    u1 = u[1]
    uu = @views u[2:M-m]
    b = [zeros(T, 2m + 1); cm]
    for k in -m:m
        b[m-k+1] = @views -dot(b[m-k+2:M-k], uu) / u1
    end
    @views b[m+2:2m+1] .+= b[m:-1:1]
    p = @views c[1:m+1] - b[m+1:2m+1]

    return postprocess(c, p, abs(s), o)
end

function polynomialcf_constant(c::AbstractVector{T}) where {T <: AbstractFloat}
    M = length(c) - 1
    D, V = @views eigs_hankel(c[2:M+1]; nev = 1)
    s, u = only(D), vec(V)
    p = @views c[1] + dot(c[2:M], u[2:M]) / u[1]
    return [p], abs(s)
end

function rationalcf(c::AbstractVector{T}, m::Int, n::Int, o::CFOptions{T}) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested numerator degree m = $(m) must be non-negative"
    @assert 0 <= n "Requested denominator degree n = $(n) must be non-negative"

    iszero(o.vscale) && return postprocess(c, zeros(T, 1), ones(T, 1), zero(T), o) # trivial function
    c = @views c[1:min(end - 1, o.maxdegree)+1] ./ o.vscale # truncate to maxdegree and normalize scale
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
        p′, q′, _, s = rationalcf(c′, m ÷ 2, n ÷ 2, CFOptions(o, c′))
        p = unzip(p′; even = true)
        q = unzip(q′; even = true)
        return postprocess(c, p, q, s, o)
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
    s, u, δ⁻, δ⁺, is_rat = eigs_hankel_block(a, m, n; atol = o.atolrat)
    if δ⁻ > 0 || δ⁺ > 0
        if is_rat
            # f is rational (at least up to machine precision)
            p, q = pade(c, m - δ⁻, n - δ⁻)
            return postprocess(c, p, q, eps(T), o)
        end

        # Try upper-left (m + δ⁺, n - δ⁻) corner of CF table block
        s′, u′, δ⁻′, δ⁺′, _ = eigs_hankel_block(a, m + δ⁺, n - δ⁻; atol = o.atolrat)
        if δ⁻′ > 0 || δ⁺′ > 0
            # Otherwise, go to lower-right (m - δ⁻, n + δ⁺) corner of CF table block
            s, u, _, _, _ = eigs_hankel_block(a, m - δ⁻, n + δ⁺; atol = o.atolrat)
            n = n + δ⁺
        else
            s, u = s′, u′
            n = n - δ⁻
        end
    end

    # Compute Chebyshev coefficients of approximation R̃(x) (eqn. 1.7b[2]) from the Laurent coefficients
    # of the Blaschke product b(z) (eqn. 1.6b[2]) via FFT
    N = max(nextpow(2, length(u)), o.minnfft)
    c̃ = zeros(T, m + 1) # Chebyshev coefficients of R̃(x): [c̃₀, c̃₁, ..., c̃ₘ] = [2*a₀, a₁ + a₋₁, ..., aₘ + a₋ₘ]
    c̃_prev = copy(c̃)
    Δc̃ = T(Inf)
    while true
        c̃_prev, c̃ = c̃, c̃_prev
        c̃_full = blaschke_laurent_coeffs(u, N, M) # Laurent coefficients of R̃(x): [a₀, a₁, ..., aₖ₋₁, a₋ₖ, ..., a₋₁] where k = N ÷ 2
        @views c̃ .= c̃_full[1:m+1]
        @views c̃[2:end] .+= c̃_full[end:-1:end-m+1]
        c̃[1] *= 2
        Δc̃_prev, Δc̃ = Δc̃, maximum(abs, c̃ - c̃_prev)
        ((N *= 2) > o.maxnfft || (Δc̃ < √o.rtolfft * maximum(abs, c̃) && Δc̃ > Δc̃_prev / 2) || isapprox(c̃, c̃_prev; rtol = o.rtolfft)) && break
    end

    @views c̃ .= a[end:-1:end-m] .- s .* c̃ # (eqn. 1.7b[2])
    s = abs(s)

    # Compute the polynomial q(z) via incomplete factorization (sec. 3.2[3]) of the denominator of the Blaschke product b(z) (eqn. 1.6b[2]).
    # This is done by computing the Laurent coefficients of p'(z) / p(z) via FFT, where p(z) is the numerator of b(z),
    # noting that the numerator of b(z) is the reciprocal polynomial zⁿ⋅b(z⁻¹) of the denominator.
    N = max(nextpow(2, length(u)), o.minnfft)
    ud = polyder(u)
    s̃ = zeros(T, n)
    s̃_prev = copy(s̃)
    Δs̃ = T(Inf)
    while true
        s̃_prev, s̃ = s̃, s̃_prev
        @views s̃ .= incomplete_poly_fact_laurent_coeffs(u, ud, N)[end-n:end-1]
        Δs̃_prev, Δs̃ = Δs̃, maximum(abs, s̃ - s̃_prev)
        ((N *= 2) > o.maxnfft || (Δs̃ < √o.rtolfft * maximum(abs, s̃) && Δs̃ > Δs̃_prev / 2) || isapprox(s̃, s̃_prev; rtol = o.rtolfft)) && break
    end

    # Compute coefficients bᵢ of the incomplete factorization q(z) = ∑ᵢ₌₀ⁿ bᵢzⁱ (eqn. 3.11[3])
    b = ones(T, n + 1)
    for j in 1:n
        b[j+1] = @views -dot(b[1:j], s̃[end-j+1:end]) / j
    end
    reverse!(b)

    # Compute the Chebyshev coefficients of the polynomial Q(x) = q(z)q(z⁻¹)/τ = |q(z)|²/τ (eqn. 1.10[2]) where z ∈ ∂D⁺ and τ = q(i)q(-i).
    q, ρ = rationalcf_normalized_denominator(b; o.quiet)
    n = length(q) - 1 # denominator degree may change in degenerate cases
    Q = Fun(ChebyshevInterval{T}(), q)

    # Compute numerator polynomial from Chebyshev expansion of 1/Q(x) and R̃(x)
    if n > 0
        # We know the exact ellipse of analyticity for 1/Q(x), so use this knowledge to
        # obtain its Chebyshev coefficients:
        ρ⁻¹ = inv(clamp(ρ, eps(T), 1 - eps(T)))
        nq⁻¹ = ceil(Int, log(4 / eps(T) / (ρ⁻¹ - 1)) / log(ρ⁻¹))
        Q⁻¹ = Fun(x -> inv(Q(x)), ChebyshevInterval{T}(), min(nq⁻¹, o.maxdegree))
        γ = zeropadresize!(coefficients(Q⁻¹), 2m + 1)
    else
        γ = zeros(T, 2m + 1)
        γ[1] = inv(q[1])
    end
    γ₀ = γ[1] *= 2
    Γ = toeplitz(γ)

    if m == 0
        p = [c̃[1] / γ₀]
        return postprocess(c, p, q, s, o)
    end

    # The following steps reduce the Toeplitz system of size 2*m + 1 to a system of
    # size m, and then solve it. If q has zeros close to the domain, then G is
    # ill-conditioned, and accuracy is lost
    A = @views Γ[1:m, 1:m]
    B = @views Γ[1:m, m+1]
    C = @views Γ[1:m, end:-1:m+2]
    G = Hermitian(@. A + C - (2 / γ₀) * (B * B'))

    if cond(G) > max(s * o.rtolcond, o.atolcond)
        !o.quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
    end

    p = zeros(T, m + 1)
    @views ldiv!(p[1:m], LinearAlgebra.cholesky!(G), (-2 * c̃[1] / γ₀) .* B .+ 2 .* c̃[m+1:-1:2])
    @views p[m+1] = (c̃[1] - dot(p[1:m], B)) / γ₀
    reverse!(p)

    return postprocess(c, p, q, s, o)
end

function rationalcf_normalized_denominator(b::AbstractVector{T}; quiet::Bool = false) where {T <: AbstractFloat}
    # Compute the Chebyshev coefficients of the polynomial Q(x) = q(z)q(z⁻¹)/τ = |q(z)|²/τ (eqn. 1.10[2]),
    # where z ∈ ∂D⁺, τ = q(i)q(-i), and q(z) = ∑ᵢ₌₀ⁿ bᵢzⁱ is "the normalized denominator of r̃∗(z) (eqn. 1.6a[2]) -
    # the polynomial of degree <= n with constant term b₀ = 1 whose zeros are the finite poles of r̃∗(z) lying outside ∂D"[2].
    # Returns the Chebyshev coefficients of Q(x) and the radius of the annulus ρ < |z| < ρ⁻¹ which contains no zeros.
    z = polyroots(b)
    ρ = maximum(abs, z)

    if ρ < 1
        # Normal case; roots are all within the unit disk. Compute Chebyshev coefficients of Q(x) by expanding the product directly:
        #   Q̃(x) = q(z)q(z⁻¹) = (∑ᵢ₌₀ⁿ bᵢ zⁱ) (∑ⱼ₌₀ⁿ bⱼ z⁻ʲ) = ∑ᵢ,ⱼ₌₀ⁿ bᵢ bⱼ zⁱ⁻ʲ
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
        normalize_chebpoly!(q) # normalize by τ = q(i)q(-i) = Q̃(0)
    else
        # Degenerate case; roots are not all within the unit disk. As in chebfun, we filter out roots with |z| > 1
        # and then compute Q(x) using the renormalized q(z).
        !quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
        filter!(zi -> abs(zi) < 1, z)
        isempty(z) && return ones(T, 1), zero(T)
        ρ = maximum(abs, z)
        n = length(z)

        # Form Q(x) using Chebyshev interpolation of q(z)q(z⁻¹)/τ; this is more numerically stable than expanding ∏ᵢ₌₁ⁿ (z - zᵢ)(z⁻¹ - zᵢ)
        @. z = (z + inv(z)) / 2
        Π = prod(-, z)
        Q = Fun(x -> real(prod(zi -> x - zi, z) / Π), ChebyshevInterval{T}(), n + 1)
        q = coefficients(Q)
    end

    return q, ρ
end

function rationalcf_reduced(c::AbstractVector, m::Int, o::CFOptions)
    p, q, _, s = polynomialcf(c, m, CFOptions(o, c))
    return postprocess(c, p, q, s, o)
end

function postprocess(c::AbstractVector{T}, p::AbstractVector{T}, s::T, o::CFOptions{T}) where {T <: AbstractFloat}
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
    s *= o.vscale

    return PolynomialApproximant(p, o.dom, s)
end

function postprocess(c::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, s::T, o::CFOptions{T}) where {T <: AbstractFloat}
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
    s *= o.vscale

    return RationalApproximant(p, q, o.dom, s)
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
    "Step size for Newton iteration"
    stepsize::T = one(T)
    "Suppress warnings"
    quiet::Bool = false
end
const MinimaxOptionsFieldnames = Symbol[fieldnames(MinimaxOptions)...]

MinimaxOptions(c::AbstractVector{T}, dom::Tuple; kwargs...) where {T <: AbstractFloat} = MinimaxOptions{T}(; dom = check_endpoints(T.(dom)), parity = parity(c), vscale = vscale(c), kwargs...)
MinimaxOptions(o::MinimaxOptions{T}, c::AbstractVector{T}) where {T <: AbstractFloat} = (@set! o.parity = parity(c); @set! o.vscale = vscale(c); return o)

function minimax(fun::Fun, m::Int, n::Int; kwargs...)
    p₀, q₀, _, _ = rationalcf(fun, m, n)
    return minimax(fun, p₀, q₀; kwargs...)
end
minimax(f, m::Int, n::Int; kwargs...) = minimax(Fun(f), m, n; kwargs...)
minimax(f, dom::Tuple, m::Int, n::Int; kwargs...) = minimax(Fun(f, Chebyshev(Interval(dom...))), m, n; kwargs...)

minimax(f, pq::AbstractVector...; kwargs...) = minimax(Fun(f), pq...; kwargs...)
minimax(f, dom::Tuple, pq::AbstractVector...; kwargs...) = minimax(Fun(f, Chebyshev(Interval(dom...))), pq...; kwargs...)
minimax(fun::Fun, pq::AbstractVector...; kwargs...) = minimax(coefficients_and_endpoints(fun)..., pq...; kwargs...)
minimax(f::AbstractVector{T}, pq::AbstractVector{T}...; kwargs...) where {T <: AbstractFloat} = minimax(f, (-one(T), one(T)), pq...; kwargs...)
minimax(f::AbstractVector{T}, dom::Tuple, pq::AbstractVector{T}...; kwargs...) where {T <: AbstractFloat} = minimax(f, pq..., MinimaxOptions(f, dom; kwargs...))

struct PolynomialMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    δf::Vector{T}
    p::Vector{T}
    Eₓ::Vector{T}
    Sₓ::Vector{T}
    δp_δε::Vector{T}
    J::Matrix{T}
    rhs::Vector{T}
    γ::T
end
function PolynomialMinimaxWorkspace(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    m = length(p) - 1
    nx = m + 2
    (o.parity === :even || o.parity === :odd) && (nx += 1) # extra root expected for even/odd symmetry
    f, p = Vector(f), Vector(p)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    Eₓ, Sₓ, δp_δε = ntuple(_ -> zeros(T, nx), 3)
    J, rhs = zeros(T, nx, m + 2), zeros(T, nx)
    return PolynomialMinimaxWorkspace(f, copy(f), p, Eₓ, Sₓ, δp_δε, J, rhs, o.stepsize)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    (length(p) <= 1) && return minimax_constant_polynomial(f)
    m = length(p) - 1
    workspace = PolynomialMinimaxWorkspace(f, p, o)

    iter = 0
    @views while true
        local (; f, δf, p, Eₓ, Sₓ, δp_δε, J, rhs, γ) = workspace
        δF = Fun(Chebyshev(), δf)
        nx = length(rhs)

        # Update difference functional
        @. δf[1:m+1] = f[1:m+1] - p
        iszero(δF) && return p, eps(T)

        # Compute new local extrema
        x, δFₓ = extremal_nodes(δF)
        nx₀ = length(x)
        nx′ = min(nx₀, nx)
        @. Eₓ[1:nx′] = abs(δFₓ[1:nx′]) # error at each root
        @. Sₓ[1:nx′] = sign(δFₓ[1:nx′]) # sign of error at each root
        εmin, εmax = extrema(Eₓ[1:nx′])
        ε, σε = (εmax + εmin) / 2, (εmax - εmin) / 2

        if nx₀ != nx
            if iter == 0
                (εmax <= 50 * eps(T) * max(vscale(f), one(T))) && return p, εmax # initial approximant is sufficiently accurate
                !o.quiet && @warn "Initial polynomial approximant is not good enough to initialize minimax fine-tuning"
            end
            !o.quiet && @warn "Found $(nx₀) local extrema for degree $(m) $(o.parity) polynomial approximant, but expected $(nx)"
            return p, T(NaN)
        end

        if σε <= max(o.rtol * ε, o.atol)
            # Minimax variance sufficiently small
            return check_signs(Sₓ; o.quiet) ? (p, ε) : (p, T(NaN))
        end

        # Newton iteration on the (linear) residual equation
        #   G(p, ε) = P(x) + ε * S(x) - F(x) = 0
        residual_jacobian!(J, x, Sₓ, m) # Jacobian of residual equation: dG/d[p; ε]
        @. rhs = δFₓ - ε * Sₓ # right hand side: -G(p, ε)
        ldiv!(δp_δε, qr!(J), rhs) # solve for Newton step: [δp; δε] = J \ rhs

        # Damped Newton step
        δp = δp_δε[1:m+1]
        δε = δp_δε[m+2]
        @. p += γ * δp

        if abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol)
            # Newton step is small enough, so we're done
            return check_signs(Sₓ; o.quiet) ? (p, ε) : (p, T(NaN))
        end

        if (iter += 1) > o.maxiter
            !o.quiet && @warn "Maximum number of iterations reached"
            return check_signs(Sₓ; o.quiet) ? (p, ε) : (p, T(NaN))
        end
    end
end

struct RationalMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    p::Vector{T}
    q::Vector{T}
    Pₓ::Vector{T}
    Qₓ::Vector{T}
    Eₓ::Vector{T}
    Sₓ::Vector{T}
    δp_δq_δε::Vector{T}
    J::Matrix{T}
    rhs::Vector{T}
    γ::T
end
function RationalMinimaxWorkspace(f::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    m, n = length(p) - 1, length(q) - 1
    nx = m + n + 2
    (o.parity === :even || o.parity === :odd) && (nx += 1) # extra root expected for even/odd symmetry
    f, p, q = Vector(f), Vector(p), Vector(q)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    normalize_chebrational!(p, q) # normalize such that Q(0) = 1
    Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε = ntuple(_ -> zeros(T, nx), 5)
    J, rhs = zeros(T, nx, m + n + 2), zeros(T, nx) # J may have more rows than columns for symmetric inputs
    return RationalMinimaxWorkspace(f, p, q, Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε, J, rhs, o.stepsize)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    (length(q) <= 1) && return minimax_reduced(f, p, o)
    (length(p) <= 1 && o.parity === :odd) && return minimax_constant_rational(f)
    m, n = length(p) - 1, length(q) - 1
    workspace = RationalMinimaxWorkspace(f, p, q, o)

    iter = 0
    @views while true
        local (; f, p, q, Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε, J, rhs, γ) = workspace
        F = Fun(Chebyshev(), f)
        P = Fun(Chebyshev(), p)
        Q = Fun(Chebyshev(), q)
        nx = length(rhs)

        # Update difference functional
        δF = (F - P / Q)::typeof(F)
        iszero(δF) && return p, q, eps(T)

        # Compute new local extrema
        x, δFₓ = extremal_nodes(δF)
        nx₀ = length(x)
        nx′ = min(nx₀, nx)
        @. Qₓ[1:nx′] = Q(x[1:nx′])
        @. Pₓ[1:nx′] = P(x[1:nx′])
        @. Eₓ[1:nx′] = abs(δFₓ[1:nx′]) # error at each root
        @. Sₓ[1:nx′] = sign(δFₓ[1:nx′]) # sign of error at each root
        εmin, εmax = extrema(Eₓ[1:nx′])
        ε, σε = (εmax + εmin) / 2, (εmax - εmin) / 2

        if nx₀ != nx
            if iter == 0
                (εmax <= 50 * eps(T) * max(vscale(f), one(T))) && return p, q, εmax # approximant is sufficiently accurate
                !o.quiet && @warn "Initial rational approximant is not good enough to initialize minimax fine-tuning"
            end
            !o.quiet && @warn "Found $(nx₀) local extrema for type $((m, n)) $(o.parity) rational approximant, but expected $(nx)"
            return p, q, T(NaN)
        end

        if σε <= max(o.rtol * ε, o.atol)
            # Minimax variance sufficiently small
            return check_signs(Sₓ; o.quiet) ? (p, q, ε) : (p, q, T(NaN))
        end

        # Newton iteration on the residual equation
        #   G(p, q, ε) = P(x) / Q(x) + ε * S(x) - F(x) = 0
        residual_jacobian!(J, x, Pₓ, Qₓ, Sₓ, m, n) # Jacobian of residual equation: dG/d[p; q; ε]
        @. rhs = δFₓ - ε * Sₓ # right hand side: -G(p, q, ε)
        ldiv!(δp_δq_δε, qr!(J), rhs) # solve for Newton step: [δp; δq; δε] = J \ rhs

        # Damped Newton step
        δp = δp_δq_δε[1:m+1]
        δq = δp_δq_δε[m+2:m+n+1]
        δε = δp_δq_δε[m+n+2]
        @. p += γ * δp
        @. q[2:n+1] += γ * δq
        normalize_chebrational!(p, q) # renormalize to Q(0) = 1

        if abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol) || maximum(abs, δq) <= max(o.rtol * maximum(abs, q), o.atol)
            # Newton step is small enough, so we're done
            return check_signs(Sₓ; o.quiet) ? (p, q, ε) : (p, q, T(NaN))
        end

        if (iter += 1) > o.maxiter
            !o.quiet && @warn "Maximum number of iterations reached"
            return check_signs(Sₓ; o.quiet) ? (p, q, ε) : (p, q, T(NaN))
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

    # First m+1 columns: dG/dPⱼ = Tⱼ(x), j = 0, ..., m
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

@views function residual_jacobian!(J::AbstractMatrix{T}, x::AbstractVector{T}, Pₓ::AbstractVector{T}, Qₓ::AbstractVector{T}, Sₓ::AbstractVector{T}, m::Int, n::Int) where {T <: AbstractFloat}
    # Compute Jacobian of residual equation:
    #   G(p, q, ε) = P(x) / Q(x) + ε * S(x) - F(x) = 0
    nx = length(x)
    @assert nx >= m + n + 2
    @assert size(J) == (nx, m + n + 2)
    @assert length(Pₓ) == length(Qₓ) == length(Sₓ) == nx

    # First m+1 columns: dG/dPⱼ = Tⱼ(x) / Q(x), j = 0, ..., m
    @. J[:, 1] = one(T) # T₀(x) = 1
    if m >= 1
        @. J[:, 2] = x # T₁(x) = x
    end
    for j in 3:m+1
        @. J[:, j] = 2x * J[:, j-1] - J[:, j-2] # Tⱼ(x) = 2x * Tⱼ₋₁(x) - Tⱼ₋₂(x)
    end
    @. J[:, 1:m+1] /= Qₓ

    # Next n columns: dG/dQⱼ = -P(x) * Tⱼ(x) / Q(x)², j = 1, ..., n
    @. J[:, m+2] = x # T₁(x) = x
    if n >= 2
        @. J[:, m+3] = 2x^2 - 1 # T₂(x) = 2x² - 1
    end
    for j in m+4:m+n+1
        @. J[:, j] = 2x * J[:, j-1] - J[:, j-2] # Tⱼ(x) = 2x * Tⱼ₋₁(x) - Tⱼ₋₂(x)
    end
    @. J[:, m+2:m+n+1] *= -Pₓ / Qₓ^2

    # Last column: dG/dε = S(x)
    @. J[:, m+n+2] = Sₓ

    return J
end

function extremal_nodes(δF::Fun)
    x₀, x₁ = endpoints(domain(δF))
    δF₀, δF₁ = δF(x₀), δF(x₁)

    # Local extremal nodes
    x = ApproxFun.roots(ApproxFun.differentiate(δF))
    isempty(x) && return [x₀; x₁], [δF₀; δF₁]
    sort!(x)

    # An endpoint is an extremal node if it has a different sign than the nearest interior extremal node
    δFₓ = δF.(x)
    if sign(δF₀) != sign(δFₓ[1])
        pushfirst!(x, x₀)
        pushfirst!(δFₓ, δF₀)
    end
    if sign(δF₁) != sign(δFₓ[end])
        push!(x, x₁)
        push!(δFₓ, δF₁)
    end

    return x, δFₓ
end

function minimax_reduced(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    p, ε = minimax(f, p, o)
    return p, ones(T, 1), T(ε)
end

function minimax_constant_polynomial(f::AbstractVector{T}) where {T <: AbstractFloat}
    lo, hi = extrema(Fun(Chebyshev(), f))
    return [T(lo + hi) / 2], T(hi - lo) / 2
end

function minimax_constant_rational(f::AbstractVector{T}) where {T <: AbstractFloat}
    p, ε = minimax_constant_polynomial(f)
    return p, ones(T, 1), T(ε)
end

function check_signs(S; quiet = false)
    pass = all(S[i] == -S[i+1] for i in 1:length(S)-1)
    !quiet && !pass && @warn "Newton iterations converged, but error signs are not alternating"
    return pass
end

#### Linear algebra utilities

function blaschke_laurent_coeffs(u::AbstractVector{T}, N::Int, M::Int) where {T <: AbstractFloat}
    # Compute Laurent coefficients of the quotient of Blaschke products:
    #   b(z) = z^M * P(z) / Q(z)
    #        = z^M * (∑_{k=0}^{d-1} u_k z^k) / (∑_{k=0}^{d-1} u_{d-k-1} z^k).
    # To efficiently evaluate b(z) for many z on the unit circle, u is padded u with
    # zeros to length N and the numerator is computed using the FFT. Additionally,
    # we save an FFT by using shift/reflect FFT identities to compute the denominator.
    d = length(u)
    û = rfft(zeropad(u, N))
    θ = (2 * (M - d + 1) // N) * (T(0):T(N ÷ 2))
    b̂ = @. cispi(-θ) * z_div_conj_z(û)
    return irfft(b̂, N)
end

function incomplete_poly_fact_laurent_coeffs(u::AbstractVector{T}, du::AbstractVector{T}, N::Int) where {T <: AbstractFloat}
    û = rfft(zeropad(u, N))
    dû = rfft(zeropad(du, N))
    return irfft(dû ./ û, N)
end

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

hankel(c::AbstractVector) = Hankel(zeropad(c, 2 * length(c) - 1))
hankel(c::AbstractVector, r::AbstractVector) = Hankel(c, r)
toeplitz(c::AbstractVector) = Toeplitz(c, c)

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

    s = D[n+1]
    u = V[:, n+1]

    δ⁻ = 0
    while δ⁻ < n && abs(abs(D[n-δ⁻]) - abs(s)) < atol
        δ⁻ += 1
    end

    δ⁺ = 0
    while n + δ⁺ + 2 < nev && abs(abs(D[n+δ⁺+2]) - abs(s)) < atol
        δ⁺ += 1
    end

    # Flag indicating if the function is actually rational
    is_rat = n + δ⁺ + 2 == nev

    return s, u, δ⁻, δ⁺, is_rat
end

#### Helper functions

function check_endpoints(dom::Tuple)
    @assert length(dom) == 2 "Domain must be a tuple of length 2"
    @assert dom[1] < dom[2] "Domain must be increasing"
    @assert all(isfinite, dom) "Domain must be finite"
    return promote(float.(dom)...)
end
check_endpoints(dom::Union{Interval, ChebyshevInterval}) = check_endpoints(endpoints(dom))
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

function chebpoly_eval_at_zero(c::AbstractVector{T}) where {T <: AbstractFloat}
    # Compute P(0) for a Chebyshev series P(x) = ∑ₖ cₖ Tₖ(x).
    p₀ = zero(T)
    sgn = +1
    for k in 1:2:length(c)
        p₀ += sgn * c[k] # Tₖ(0) = {+1 if k ≡ 0 (mod 4), -1 if k ≡ 2 (mod 4)}
        sgn = -sgn
    end
    return p₀
end
normalize_chebpoly!(p::AbstractVector) = p ./= chebpoly_eval_at_zero(p)
normalize_chebpoly(p::AbstractVector) = normalize_chebpoly!(copy(p))
normalize_chebrational!(p::AbstractVector, q::AbstractVector) = scale_rational!(p, q, chebpoly_eval_at_zero(q))
normalize_chebrational(p::AbstractVector, q::AbstractVector) = normalize_chebrational!(copy(p), copy(q))

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
function lazychopcoeffs(c::AbstractVector{T}; rtol::T = eps(T), atol::T = zero(T), parity = :generic) where {T <: AbstractFloat}
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

end # module CaratheodoryFejerApprox
