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

using ApproxFun: Chebyshev, Fun, Interval, coefficients, domain, endpoints, ncoefficients, space
using Arpack: eigs
using FFTW: rfft, irfft
using GenericFFT: GenericFFT # defines generic methods for FFTs
using GenericLinearAlgebra: GenericLinearAlgebra # defines generic method for `LinearAlgebra.eigen` and `Polynomials.roots`
using LinearAlgebra: Hermitian, cond, dot, eigen
using Polynomials: Polynomials, Polynomial

export polynomialcf, rationalcf

function polynomialcf(fun::Fun, m::Int, M::Int = ncoefficients(fun) - 1)
    @assert M + 1 <= ncoefficients(fun) "Requested number of Chebyshev expansion coefficients $(M) exceeds the number of coefficients $(ncoefficients(fun)) of the function"
    fun = check_fun(fun)
    c = coefficients(fun)[1:M+1]
    if m >= M
        return c, zero(eltype(c))
    else
        return polynomialcf(c, m)
    end
end

polynomialcf(f, dom::Interval, m::Int) = polynomialcf(Fun(f, Chebyshev(dom)), m)
polynomialcf(f, dom::Tuple, m::Int) = polynomialcf(f, Interval(dom...), m)

function rationalcf(fun::Fun, m::Int, n::Int, M::Int = ncoefficients(fun) - 1; kwargs...)
    @assert M + 1 <= ncoefficients(fun) "Requested number of Chebyshev expansion coefficients $(M) exceeds the number of coefficients $(ncoefficients(fun)) of the function"
    fun = check_fun(fun)
    c = coefficients(fun)[1:M+1]
    if m >= M
        return c, [one(eltype(c))], zero(eltype(c))
    else
        return rationalcf(c, m, n; kwargs...)
    end
end

rationalcf(f, dom::Interval, m::Int, n::Int; kwargs...) = rationalcf(Fun(f, Chebyshev(dom)), m, n; kwargs...)
rationalcf(f, dom::Tuple, m::Int, n::Int; kwargs...) = rationalcf(f, Interval(dom...), m, n; kwargs...)

function polynomialcf(c::AbstractVector{T}, m::Int) where {T <: AbstractFloat}
    @assert m <= length(c)
    M = length(c) - 1

    # Trivial case
    (m == M - 1) && return c[1:M], abs(c[M+1])

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

    return p, s
end

function rationalcf(
    c::AbstractVector{T},
    m::Int,
    n::Int;
    atolrat = 50 * eps(T),      # Tolerance for detecting rationality
    rtolfft = eps(T)^(2 // 3),  # Relative tolerance for FFT deconvolution
    minnfft = 2^8,              # Minimum FFT length
    maxnfft = 2^17,             # Maximum FFT length
    rtolcond = 1e13,            # Relative tolerance for detecting ill-conditioning
    atolcond = 1e3,             # Absolute tolerance for detecting ill-conditioning
    quiet = false,              # Suppress warnings
) where {T <: AbstractFloat}
    @assert m <= length(c)
    n == 0 && return rationalcf_trivial(c, m)

    M = length(c) - 1
    vscale = sum(abs, c) # crude bound on infinity norm of Fun(Chebyshev(), c)

    # Reorder coeffs and scale T_0 coefficient
    a = copy(c)
    a[1] *= 2
    reverse!(a)

    # Check even / odd symmetries
    if maximum(abs, @views a[end-1:-2:1]) < vscale * eps(T) # f is even
        if iseven(m) && iseven(n)
            m += 1
        elseif isodd(m) && isodd(n)
            (n -= 1) == 0 && return rationalcf_trivial(c, m)
        end
    elseif maximum(abs, @views a[end:-2:1]) < vscale * eps(T) # f is odd
        if isodd(m) && iseven(n)
            m += 1
        elseif iseven(m) && isodd(n)
            (n -= 1) == 0 && return rationalcf_trivial(c, m)
        end
    end

    # Obtain eigenvalues and block structure
    s, u, k, l, rflag = eigs_hankel_block(a, m, n; tol = atolrat)
    if k > 0 || l > 0
        if rflag
            # f is rational (at least up to machine precision)
            p, q = pade(c, m - k, n - k)
            return p, q, eps(T)
        end

        nnew = n - k
        s, u, knew, lnew, _ = eigs_hankel_block(a, m + l, nnew; tol = atolrat)
        if knew > 0 || lnew > 0
            n = n + l
            s, u, k, l, _ = eigs_hankel_block(a, m - k, n; tol = atolrat)
        else
            n = nnew
        end
    end

    # Obtain polynomial q from Laurent coefficients using FFT
    N = max(nextpow(2, length(u)), minnfft)
    ud = polyder(u)
    ac = deconv(ud, u, N)
    while true
        N *= 2
        ac_last = ac
        ac = deconv(ud, u, N)
        diff = @views reldiff(ac[end-n:end-1], ac_last[end-n:end-1])
        (diff <= rtolfft || N >= maxnfft) && break
    end

    b = ones(T, n + 1)
    for j in 1:n
        b[j+1] = @views -dot(b[1:j], ac[end-j:end-1]) / j
    end
    reverse!(b)

    z = polyroots(b)
    zmax = maximum(abs, z)
    if zmax > 1
        !quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
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
    N = max(nextpow(2, length(u)), minnfft)
    v = reverse(u)

    ac = deconv(u, v, N, M)
    while true
        N *= 2
        ac_last = ac
        ac = deconv(u, v, N, M)
        diff1 = @views reldiff(ac[1:m+1], ac_last[1:m+1])
        diff2 = @views m == 0 ? T(Inf) : reldiff(ac[end-m+1:end], ac_last[end-m+1:end])
        (diff1 <= rtolfft || diff2 <= rtolfft || N >= maxnfft) && break
    end

    ac .*= s
    ct = @views a[end:-1:end-m] .- ac[1:m+1] .- [ac[1]; ac[end:-1:end-m+1]]
    s = abs(s)

    # Compute numerator polynomial from Chebyshev expansion of 1/q and Rt. We
    # know the exact ellipse of analyticity for 1/q, so use this knowledge to
    # obtain its Chebyshev coefficients (see line below)
    nq⁻¹ = ceil(Int, log(4 / eps(T) / (rho - 1)) / log(rho))
    qfun⁻¹ = Fun(x -> inv(qfun(x)), Interval(-one(T), one(T)), nq⁻¹)
    γ = coefficients(qfun⁻¹)
    γ₀ = γ[1] *= 2
    Γ = toeplitz(cropto(zeropad(γ, 2m + 1), 2m + 1))

    if m == 0
        p = [ct[1] / γ₀]
        return p, q, s
    end

    # The following steps reduce the Toeplitz system of size 2*m + 1 to a system of
    # size m, and then solve it. If q has zeros close to the domain, then G is
    # ill-conditioned, and accuracy is lost
    A = @views Γ[1:m, 1:m]
    B = @views Γ[1:m, m+1]
    C = @views Γ[1:m, end:-1:m+2]
    G = Hermitian(A + C - (2 / γ₀) * (B * B'))

    if cond(G) > max(s * rtolcond, atolcond)
        !quiet && @warn "Ill-conditioning detected. Results may be inaccurate"
    end

    bc = @views G \ (-2 * ((ct[1] / γ₀) * B - ct[m+1:-1:2]))
    bc₀ = (ct[1] - dot(B, bc)) / γ₀
    p = @views [bc₀; bc[end:-1:1]]

    return p, q, s
end

function rationalcf_trivial(c::AbstractVector, m::Int)
    p, s = polynomialcf(c, m)
    return p, [one(eltype(c))], abs(s)
end

function pade(c::AbstractVector{T}, m, n) where {T <: AbstractFloat}
    M = length(c) - 1
    l = max(m, n) # temporary degree variable in case m < n

    # Get the Chebyshev coefficients and pad if necessary
    if M < m + 2n
        # Chebfun implementation notes that using random values may be more stable than using zeros?
        # This doesn't seem worth the non-determinism to me
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
    α = @views conv(a[1:l+1], β)[1:l+1]

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

#### Linear algebra utilities

function deconv(a::AbstractVector, b::AbstractVector, N::Int, M::Union{Int, Nothing} = nothing)
    â, b̂ = rfft(zeropad(float(a), N)), rfft(zeropad(float(b), N))
    if M === nothing
        ĉ = @. â / b̂
    else
        T = typeof(one(eltype(a)) / one(eltype(b)))
        θ = (2M // N) * (T(0):T(N ÷ 2))
        ĉ = @. cispi(-θ) * â / b̂
    end
    return irfft(ĉ, N)
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

# Build a Hankel matrix from the given vector of coefficients
function hankel(c::AbstractVector{T}) where {T}
    n = length(c)
    H = zeros(T, n, n)
    for j in 1:n, i in 1:n-j+1
        H[i, j] = c[i+j-1]
    end
    return Hermitian(H)
end

function hankel(c::AbstractVector{T}, r::AbstractVector{T}) where {T}
    @assert length(c) == length(r)
    n = length(c)
    H = zeros(T, n, n)
    for j in 1:n
        for i in 1:n-j+1
            H[i, j] = c[i+j-1]
        end
        for i in n-j+2:n
            H[i, j] = r[i+j-n]
        end
    end
    return Hermitian(H)
end

# Build a Toeplitz matrix from the given vector of coefficients
function toeplitz(c::AbstractVector{T}) where {T}
    n = length(c)
    A = zeros(T, n, n)
    for j in 1:n, i in 1:n
        A[i, j] = c[abs(i - j)+1]
    end
    return Hermitian(A)
end

function eigs_hankel(c::AbstractVector{T}; nev = 1) where {T <: AbstractFloat}
    # Declaring variable types helps inference since `eigs` is not type-stable,
    # and `eigen` may not be type-stable for e.g. BigFloat
    local D::Vector{T}, V::Matrix{T}
    m = length(c)
    A = hankel(c)

    if m >= 32 && T <: Union{Float32, Float64}
        D, V = eigs(A; nev = nev, which = :LM, v0 = ones(T, m) ./ m)
        return convert(Vector{T}, D), convert(Matrix{T}, V)
    else
        D, V = eigen(A)
        I = partialsortperm(D, 1:nev; by = abs, rev = true)
        return D[I], V[:, I]
    end
end

function eigs_hankel_block(a::AbstractVector{T}, m, n; tol = 50 * eps(T)) where {T <: AbstractFloat}
    # Each Hankel matrix corresponds to one diagonal m - n = const in the CF-table;
    # when a diagonal intersects a square block, the eigenvalues on the
    # intersection are all equal. k and l tell you how many entries on the
    # intersection appear before and after the eigenvalues under consideration.
    # u is the corresponding eigenvector
    M = length(a) - 1

    if n > M + m + 1
        c = [zeros(T, n - (M + m + 1)); a[(M+1).-abs.(-M:M)]]
    else
        c = a[(M+1).-abs.(m-n+1:M)]
    end

    nev = min(n + 10, length(c))
    D, V = eigs_hankel(c; nev = nev)

    s = D[n+1]
    u = V[:, n+1]

    k = 0
    while k < n && abs(D[n-k] - abs(s)) < tol
        k += 1
    end

    l = 0
    while n + l + 2 < nev && abs(D[n+l+2] - abs(s)) < tol
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
check_endpoints(dom::Interval) = check_endpoints(endpoints(dom))
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

reldiff(x::Number, y::Number) = abs((x - y) / x)
reldiff(x::AbstractVector, y::AbstractVector) = mapreduce(reldiff, max, x, y)

#### Polynomial utilities

function poly(c::AbstractVector{T}, dom::Union{NTuple{2, T}, Nothing} = nothing; transplant = dom !== nothing) where {T}
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

end # module CaratheodoryFejerApprox
