#=
CaratheodoryFejerApprox.jl is based largely on the following files from Chebfun v5:
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
using LinearAlgebra: LinearAlgebra, Hermitian, cond, diag, dot, eigen, ldiv!, mul!, qr!
using Polynomials: Polynomials, Polynomial
using Setfield: Setfield, @set
using Statistics: Statistics, mean, std
using ToeplitzMatrices: ToeplitzMatrices, Toeplitz, Hankel

export polynomialcf, rationalcf

Base.@kwdef struct CFOptions{T <: AbstractFloat}
    "Even/odd parity of f(x)"
    parity::Symbol = :generic
    "Maximum Chebyshev series degree. Series is truncated to this degree if necessary"
    maxdegree::Int = 2^10
    "Relative tolerance for chopping off small coefficients"
    rtolchop::T = eps(T)
    "Absolute tolerance for chopping off small coefficients"
    atolchop::T = zero(T)
    "Tolerance for detecting rationality"
    atolrat::T = 50 * eps(T)
    "Relative tolerance for FFT deconvolution"
    rtolfft::T = 50 * eps(T)
    "Minimum FFT length"
    minnfft::Int = 2^8
    "Maximum FFT length"
    maxnfft::Int = 2^20
    "Relative tolerance for detecting ill-conditioning"
    rtolcond::T = 1e13
    "Absolute tolerance for detecting ill-conditioning"
    atolcond::T = 1e3
    "Suppress warnings"
    quiet::Bool = false
end

CFOptions(o::CFOptions{T}, c::AbstractVector{T}) where {T <: AbstractFloat} = @set o.parity = parity(c)
CFOptions(c::AbstractVector{T}; kwargs...) where {T <: AbstractFloat} = CFOptions{T}(; parity = parity(c), kwargs...)

polynomialcf(fun::Fun, m::Int; kwargs...) = polynomialcf(coefficients(check_fun(fun)), m; kwargs...)
polynomialcf(c::AbstractVector{<:AbstractFloat}, m::Int; kwargs...) = polynomialcf(c, m, CFOptions(c; kwargs...))

polynomialcf(f, dom::Tuple, m::Int; kwargs...) = polynomialcf(Fun(f, Chebyshev(Interval(dom...))), m; kwargs...)
polynomialcf(f, m::Int; kwargs...) = polynomialcf(Fun(f), m; kwargs...)

rationalcf(fun::Fun, m::Int, n::Int; kwargs...) = rationalcf(coefficients(check_fun(fun)), m, n; kwargs...)
rationalcf(c::AbstractVector{<:AbstractFloat}, m::Int, n::Int; kwargs...) = rationalcf(c, m, n, CFOptions(c; kwargs...))

rationalcf(f, dom::Tuple, m::Int, n::Int; kwargs...) = rationalcf(Fun(f, Chebyshev(Interval(dom...))), m, n; kwargs...)
rationalcf(f, m::Int, n::Int; kwargs...) = rationalcf(Fun(f), m, n; kwargs...)

function polynomialcf(c::AbstractVector{T}, m::Int, o::CFOptions) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested polynomial degree m = $(m) must be non-negative"

    c = @views c[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
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
    m == M && return cleanup_coeffs(c[1:M+1], o), zero(T)
    m == M - 1 && return cleanup_coeffs(c[1:M], o), abs(c[M+1])

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

    return cleanup_coeffs(p, o), abs(s)
end

function rationalcf(c::AbstractVector{T}, m::Int, n::Int, o::CFOptions{T}) where {T <: AbstractFloat}
    @assert !isempty(c) "Chebyshev coefficient vector must be non-empty"
    @assert 0 <= m "Requested numerator degree m = $(m) must be non-negative"
    @assert 0 <= n "Requested denominator degree n = $(n) must be non-negative"

    c = @views c[1:min(end - 1, o.maxdegree)+1] # view of coefficients truncated to `maxdegree`
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
    s, u, k, l, rflag = eigs_hankel_block(a, m, n; atol = o.atolrat)
    if k > 0 || l > 0
        if rflag
            # f is rational (at least up to machine precision)
            p, q = pade(c, m - k, n - k)
            return cleanup_coeffs(p, q, o)..., eps(T)
        end

        n′ = n - k
        s, u, k′, l′, _ = eigs_hankel_block(a, m + l, n′; atol = o.atolrat)
        if k′ > 0 || l′ > 0
            n = n + l
            s, u, k, l, _ = eigs_hankel_block(a, m - k, n; atol = o.atolrat)
        else
            n = n′
        end
    end

    # Obtain polynomial q from Laurent coefficients using FFT
    N = max(nextpow(2, length(u)), o.minnfft)
    ud = polyder(u)
    bc = zeros(T, n)
    bc_prev = copy(bc)
    Δbc = T(Inf)
    while true
        bc_prev, bc = bc, bc_prev
        @views bc .= incomplete_poly_fact_laurent_coeffs(u, ud, N)[end-n:end-1]
        Δbc_prev, Δbc = Δbc, maximum(abs, bc - bc_prev)
        ((N *= 2) > o.maxnfft || (Δbc < √o.rtolfft * maximum(abs, bc) && Δbc > Δbc_prev / 2) || isapprox(bc, bc_prev; rtol = o.rtolfft)) && break
    end

    b = ones(T, n + 1)
    for j in 1:n
        b[j+1] = @views -dot(b[1:j], bc[end-j+1:end]) / j
    end
    reverse!(b)

    z = polyroots(b)
    zmax = maximum(abs, z)
    if zmax > 1
        !o.quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
        filter!(zi -> abs(zi) < 1, z)
        zmax = maximum(abs, z)
    end

    rho = inv(zmax)
    z = @. (z + inv(z)) / 2

    # Compute q from the roots for stability reasons
    Π = prod(-, z)
    qfun = Fun(x -> real(prod(zi -> x - zi, z) / Π), Interval(-one(T), one(T)))
    q = coefficients(qfun)

    # Compute Chebyshev coefficients of approximation Rt from Laurent coefficients
    # of Blaschke product using FFT
    N = max(nextpow(2, length(u)), o.minnfft)
    ac = zeros(T, m + 1) # symmetrized Laurent coefficients of Rt: [2*a₀, a₁ + a₋₁, ..., aₘ + a₋ₘ]
    ac_prev = copy(ac)
    Δac = T(Inf)
    while true
        ac_prev, ac = ac, ac_prev
        ac_full = blaschke_laurent_coeffs(u, N, M) # Laurent coefficients of Rt: [a₀, a₁, ..., aₖ₋₁, a₋ₖ, ..., a₋₁] where k = N ÷ 2
        @views ac .= ac_full[1:m+1]
        @views ac[2:end] .+= ac_full[end:-1:end-m+1]
        ac[1] *= 2
        Δac_prev, Δac = Δac, maximum(abs, ac - ac_prev)
        ((N *= 2) > o.maxnfft || (Δac < √o.rtolfft * maximum(abs, ac) && Δac > Δac_prev / 2) || isapprox(ac, ac_prev; rtol = o.rtolfft)) && break
    end

    ct = @views a[end:-1:end-m] .- s .* ac # (eqn. 1.7b)
    s = abs(s)

    # Compute numerator polynomial from Chebyshev expansion of 1/q and Rt. We
    # know the exact ellipse of analyticity for 1/q, so use this knowledge to
    # obtain its Chebyshev coefficients (see line below)
    nq⁻¹ = ceil(Int, log(4 / eps(T) / (rho - 1)) / log(rho))
    qfun⁻¹ = Fun(x -> inv(qfun(x)), Interval(-one(T), one(T)), nq⁻¹)
    γ = coefficients(qfun⁻¹)
    γ = cropto(zeropad(γ, 2m + 1), 2m + 1)
    γ₀ = γ[1] *= 2
    Γ = toeplitz(γ)

    if m == 0
        p = [ct[1] / γ₀]
        return cleanup_coeffs(p, q, o)..., s
    end

    # The following steps reduce the Toeplitz system of size 2*m + 1 to a system of
    # size m, and then solve it. If q has zeros close to the domain, then G is
    # ill-conditioned, and accuracy is lost
    A = @views Γ[1:m, 1:m]
    B = @views Γ[1:m, m+1]
    C = @views Γ[1:m, end:-1:m+2]
    G = Hermitian(A + C - (2 / γ₀) * (B * B'))

    if cond(G) > max(s * o.rtolcond, o.atolcond)
        !o.quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
    end

    bc = @views G \ (-2 * ((ct[1] / γ₀) * B - ct[m+1:-1:2]))
    bc₀ = (ct[1] - dot(B, bc)) / γ₀
    p = @views [bc₀; bc[end:-1:1]]

    return cleanup_coeffs(p, q, o)..., s
end

function rationalcf_reduced(c::AbstractVector, m::Int, o::CFOptions)
    p, s = polynomialcf(c, m, o)
    return p, [one(eltype(c))], abs(s)
end

function cleanup_coeffs(p::AbstractVector{T}, o::CFOptions{T}) where {T <: AbstractFloat}
    # Clean up even/odd symmetric coefficients
    p = paritychop(p, o.parity)
    if o.parity === :even
        @views p[2:2:end] .= zero(T)
    elseif o.parity === :odd
        @views p[1:2:end] .= zero(T)
    end
    return p
end

function cleanup_coeffs(p::AbstractVector{T}, q::AbstractVector{T}, o::CFOptions{T}) where {T <: AbstractFloat}
    # Clean up even/odd symmetric coefficients
    if o.parity === :even
        p, q = paritychop(p, :even), paritychop(q, :even)
        @views p[2:2:end] .= zero(T) # even
        @views q[2:2:end] .= zero(T) # even
    elseif o.parity === :odd
        p, q = paritychop(p, :odd), paritychop(q, :even)
        @views p[1:2:end] .= zero(T) # odd
        @views q[2:2:end] .= zero(T) # even
    end
    return p, q
end

#### Pade

function pade(c::AbstractVector{T}, m, n) where {T <: AbstractFloat}
    M = length(c) - 1
    l = max(m, n) # temporary degree variable in case m < n

    # Get the Chebyshev coefficients and pad if necessary
    if M < m + 2n
        # Chebfun implementation notes that using random values may be more stable than using zeros?
        # This doesn't seem worth the non-determinism to me...
        # a = [c; eps(T) * randn(T, m + 2n - M)]
        a = [c; zeros(T, m + 2n + M, 1)]
    else
        a = copy(c)
    end
    a[1] *= 2

    # Set up and solve Hankel system for denominator Laurent-Pade coefficients
    top = @views a[abs.(m-n+1:m).+1]  # Top row of Hankel system
    bot = @views a[m+1:m+n]           # Bottom row of Hankel system
    rhs = @views a[m+2:m+n+1]         # RHS of Hankel system

    if n > 0
        β = [one(T); -reverse!(hankel(top, bot) \ rhs)]
    else
        β = [one(T)]
    end

    # Use convolution to compute numerator Laurent-Pade coefficients
    a[1] /= 2
    α = @views conv(a[1:l+1], β)[1:l+1] # Note: direct linear convolution is faster than FFT, since l = max(m, n) will never be very large in practice

    # Compute numerator Chebyshev-Pade coefficients
    p = zeros(T, m + 1)
    D = zeros(T, l + 1, l + 1)
    @views D[1:l+1, 1:n+1] .= α .* β'
    for j in 1:l+1, i in max(1, j - m):min(l + 1, j + m)
        p[abs(i - j)+1] += D[i, j] # p[1] = sum(diag(D)) and p[k+1] = sum(diag(D, k)) + sum(diag(D, -k)) for 1 <= k <= m
    end

    # Compute denominator Chebyshev-Pade coefficients
    q = zeros(T, n + 1)
    for k in 1:n+1
        u = @views β[1:n+2-k]
        v = @views β[k:end]
        q[k] = dot(u, v)
    end

    # Normalize the coefficients
    p ./= q[1]
    q ./= q[1] / 2
    q[1] = one(T)

    return p, q
end

#### Minimax

Base.@kwdef struct MinimaxOptions{T <: AbstractFloat}
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
end

function minimax(fun::Fun, m::Int, n::Int; kwargs...)
    p₀, q₀, _ = rationalcf(fun, m, n)
    p, q, ε = minimax(fun, p₀, q₀; kwargs...)
    return p, q, ε
end
minimax(fun::Fun, pq::AbstractVector...; kwargs...) = minimax(coefficients(fun), pq...; kwargs...)
minimax(f::AbstractVector{T}, pq::AbstractVector{T}...; kwargs...) where {T <: AbstractFloat} = minimax(f, pq..., MinimaxOptions{T}(; kwargs...))

minimax(f, dom::Tuple, m::Int, n::Int; kwargs...) = minimax(Fun(f, Chebyshev(Interval(dom...))), m, n; kwargs...)
minimax(f, m::Int, n::Int; kwargs...) = minimax(Fun(f), m, n; kwargs...)

struct PolynomialMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    δf::Vector{T}
    p::Vector{T}
    x::Vector{T}
    δFₓ::Vector{T}
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
    f, p = collect(f), collect(p)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    x, δFₓ, Eₓ, Sₓ, δp_δε = ntuple(_ -> zeros(T, nx), 5)
    J, rhs = zeros(T, nx, nx), zeros(T, nx)
    x[begin] = -one(T)
    x[end] = one(T)
    return PolynomialMinimaxWorkspace(f, copy(f), p, x, δFₓ, Eₓ, Sₓ, δp_δε, J, rhs, o.stepsize)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    if length(p) <= 1
        lo, hi = extrema(Fun(Chebyshev(), f))
        return [(lo + hi) / 2], (hi - lo) / 2
    end
    m = length(p) - 1
    workspace = PolynomialMinimaxWorkspace(f, p, o)

    iter = 0
    @views while true
        local (; f, δf, p, x, δFₓ, Eₓ, Sₓ, δp_δε, J, rhs, γ) = workspace
        δF = Fun(Chebyshev(), δf)

        # Compute new local extrema
        @. δf[1:m+1] = f[1:m+1] - p
        x₀ = ApproxFun.roots(ApproxFun.differentiate(δF))
        nx₀ = length(x₀)
        if nx₀ != m
            iter == 0 && @warn "Initial polynomial is not a good enough approximant for initializing minimax"
            @warn "Found $(nx₀) local extrema, but expected $(m)"
            return p, T(NaN)
        end

        @. x[2:m+1] = x₀
        @. δFₓ = δF(x) # Note: (F - P)(x) is much more accurate than F(x) - P(x)
        @. Eₓ = abs(δFₓ) # error at each root
        @. Sₓ = sign(δFₓ) # sign of error at each root

        ε, σε = mean(Eₓ), std(Eₓ; corrected = false)
        if σε <= max(o.rtol * ε, o.atol)
            check_signs(Sₓ)
            return p, ε
        end

        # Newton iteration on the (linear) residual equation
        #   G(P, ε) = P(x) + ε * S(x) - F(x) = 0

        # First m+1 columns: dG/dPⱼ = Tⱼ(x) / Q(x), j = 0, ..., m
        @. J[:, 1] = one(T)
        if m >= 1
            @. J[:, 2] = x
        end
        for j in 3:m+1
            @. J[:, j] = 2x * J[:, j-1] - J[:, j-2]
        end

        # Last column: dG/dε = S(x)
        @. J[:, m+2] = Sₓ

        # Right-hand side: rhs = -G(P, ε)
        @. rhs = δFₓ - ε * Sₓ

        # Solve linear system: [δp; δε] = J \ rhs
        ldiv!(δp_δε, qr!(J), rhs)
        δp = δp_δε[1:m+1]
        δε = δp_δε[m+2]

        # Damped Newton step
        @. p += γ * δp

        if abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol)
            # Newton step is small enough, so we're done
            check_signs(Sₓ)
            return p, ε
        end

        if (iter += 1) > o.maxiter
            @warn "Maximum number of iterations reached"
            check_signs(Sₓ)
            return p, ε
        end
    end
end

struct RationalMinimaxWorkspace{T <: AbstractFloat}
    f::Vector{T}
    p::Vector{T}
    q::Vector{T}
    x::Vector{T}
    δFₓ::Vector{T}
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
    f, p, q = collect(f), collect(p), collect(q)
    f = f[1:min(end - 1, o.maxdegree)+1] # truncate to maxdegree
    p ./= q[1]
    q ./= q[1]
    x, δFₓ, Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε = ntuple(_ -> zeros(T, nx), 9)
    J, rhs = zeros(T, nx, nx), zeros(T, nx)
    x[begin] = -one(T)
    x[end] = one(T)
    return RationalMinimaxWorkspace(f, p, q, x, δFₓ, Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε, J, rhs, o.stepsize)
end

function minimax(f::AbstractVector{T}, p::AbstractVector{T}, q::AbstractVector{T}, o::MinimaxOptions{T}) where {T <: AbstractFloat}
    if length(q) <= 1
        p, ε = minimax(f, p, o)
        return p, [one(T)], ε
    end
    m, n = length(p) - 1, length(q) - 1
    workspace = RationalMinimaxWorkspace(f, p, q, o)

    iter = 0
    @views while true
        local (; f, p, q, x, δFₓ, Pₓ, Qₓ, Eₓ, Sₓ, δp_δq_δε, J, rhs, γ) = workspace
        F = Fun(Chebyshev(), f)
        P = Fun(Chebyshev(), p)
        Q = Fun(Chebyshev(), q)

        # Compute new local extrema
        δF = (F - P / Q)::typeof(F)
        x₀ = ApproxFun.roots(ApproxFun.differentiate(δF))
        nx₀ = length(x₀)
        if nx₀ != m + n
            iter == 0 && @warn "Initial polynomial is not a good enough approximant for initializing minimax"
            @warn "Found $(nx₀) local extrema, but expected $(m+n)"
            return p, q, T(NaN)
        end

        @. x[2:m+n+1] = x₀
        @. δFₓ = δF(x) # Note: (F - P)(x) is much more accurate than F(x) - P(x)
        @. Qₓ = Q(x)
        @. Pₓ = P(x)
        @. Eₓ = abs(δFₓ) # error at each root
        @. Sₓ = sign(δFₓ) # sign of error at each root

        ε, σε = mean(Eₓ), std(Eₓ; corrected = false)
        if σε <= max(o.rtol * ε, o.atol)
            check_signs(Sₓ)
            return p, q, ε
        end

        # Newton iteration on the residual equation
        #   G(P, Q, ε) = P(x) / Q(x) + ε * S(x) - F(x) = 0

        # First m+1 columns: dG/dPⱼ = Tⱼ(x) / Q(x), j = 0, ..., m
        @. J[:, 1] = one(T)
        if m >= 1
            @. J[:, 2] = x
        end
        for j in 3:m+1
            @. J[:, j] = 2x * J[:, j-1] - J[:, j-2]
        end
        @. J[:, 1:m+1] /= Qₓ

        # Next n columns: dG/dQⱼ = -P(x) * Tⱼ(x) / Q(x)², j = 1, ..., n
        @. J[:, m+2] = x
        if n >= 2
            @. J[:, m+3] = 2x^2 - 1
        end
        for j in m+4:m+n+1
            @. J[:, j] = 2x * J[:, j-1] - J[:, j-2]
        end
        @. J[:, m+2:m+n+1] *= -Pₓ / Qₓ^2

        # Last column: dG/dε = S(x)
        @. J[:, m+n+2] = Sₓ

        # Right hand side: -G(P, Q, ε)
        @. rhs = δFₓ - ε * Sₓ

        # Solve for Newton step: [δp; δq; δε] = J \ rhs
        ldiv!(δp_δq_δε, qr!(J), rhs)
        δp = δp_δq_δε[1:m+1]
        δq = δp_δq_δε[m+2:m+n+1]
        δε = δp_δq_δε[m+n+2]

        # Damped Newton step
        @. p += γ * δp
        @. q[2:n+1] += γ * δq

        if abs(δε) <= max(o.rtol * ε, o.atol) || maximum(abs, δp) <= max(o.rtol * maximum(abs, p), o.atol) || maximum(abs, δq) <= max(o.rtol * maximum(abs, q), o.atol)
            # Newton step is small enough, so we're done
            check_signs(Sₓ)
            return p, q, ε
        end

        if (iter += 1) > o.maxiter
            @warn "Maximum number of iterations reached"
            check_signs(Sₓ)
            return p, q, ε
        end
    end
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
    # Efficiently and safely compute z / conj(z). Returns 1 if z == 0.
    a, b = reim(z)
    r² = a^2 + b^2
    return ifelse(r² == 0, one(z), Complex((a - b) * (a + b), 2 * a * b) / r²)
end

function conv(a::AbstractVector, b::AbstractVector)
    N, M = length(a), length(b)
    T = typeof(one(eltype(a)) * one(eltype(b)))
    c = zeros(T, N + M - 1)
    for i in 1:N, j in 1:M
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

    k = 0
    while k < n && abs(abs(D[n-k]) - abs(s)) < atol
        k += 1
    end

    l = 0
    while n + l + 2 < nev && abs(abs(D[n+l+2]) - abs(s)) < atol
        l += 1
    end

    # Flag indicating if the function is actually rational
    rFlag = n + l + 2 == nev

    return s, u, k, l, rFlag
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

function zeropad(x::AbstractVector, n::Int)
    return n <= length(x) ? x : [x; zeros(eltype(x), n - length(x))]
end

function cropto(x::AbstractVector, n::Int)
    return n >= length(x) ? x : x[1:n]
end

#### Polynomial utilities

function poly(c::AbstractVector{T}, dom::Union{NTuple{2, T}, Nothing} = nothing; transplant = dom !== nothing) where {T}
    # Convert a vector of Chebyshev coefficients to a vector of monomial coefficients,
    # optionally linearly transplanted to a new domain.
    @assert !(transplant && dom === nothing) "Domain must be specified when transplanting"
    Q = Polynomials.ChebyshevT{T, :x}(c) # polynomial in Chebyshev basis on [-1, 1]
    P = convert(Polynomial{T, :x}, Q) # polynomial in monomial basis on [-1, 1]
    if transplant
        a, b = dom
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

function parity(c::AbstractVector{T}; rtol = eps(T)) where {T <: AbstractFloat}
    length(c) <= 1 && return :even # constant function
    scale = vscale(c)
    scale == zero(T) && return :even
    oddscale = @views vscale(c[1:2:end])
    oddscale <= rtol * scale && return :odd
    evenscale = @views vscale(c[2:2:end])
    evenscale <= rtol * scale && return :even
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
chopcoeffs(c::AbstractVector; kwargs...) = collect(lazychopcoeffs(c; kwargs...))
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
paritychop(c::AbstractVector, parity::Symbol) = collect(lazyparitychop(c, parity))

# Crude bound on infinity norm of `fun`
vscale(fun::Fun) = vscale(coefficients(fun))
vscale(c::AbstractVector{T}) where {T <: AbstractFloat} = sum(abs, c; init = zero(T))

end # module CaratheodoryFejerApprox
