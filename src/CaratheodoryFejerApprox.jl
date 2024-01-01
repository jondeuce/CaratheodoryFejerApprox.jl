module CaratheodoryFejerApprox

using ApproxFun: Chebyshev, Fun, Interval, Taylor, coefficients, domain, endpoints, extrapolate, ncoefficients, space
using Arpack: eigs
using LinearAlgebra: Symmetric, dot, eigen

export polynomialcf

function check_endpoints(dom::Tuple)
    @assert length(dom) == 2 "Domain must be a tuple of length 2"
    @assert dom[1] < dom[2] "Domain must be increasing"
    @assert all(isfinite, dom) "Domain must be finite"
    return promote(float.(dom)...)
end

function check_fun(fun::Fun)
    @assert isreal(fun) "Only real-valued functions are supported"
    @assert space(fun) isa Chebyshev "Only Chebyshev spaces are supported"
    check_endpoints(endpoints(domain(fun)))
    return fun
end

function polynomialcf(fun::Fun, m::Int, M::Int = ncoefficients(fun) - 1)
    @assert M + 1 <= ncoefficients(fun) "Requested number of Chebyshev expansion coefficients $(M) exceeds the number of coefficients $(ncoefficients(fun)) of the function"
    fun = check_fun(fun)
    a = coefficients(fun)[1:M+1]
    if m >= M
        return a, zero(eltype(a))
    else
        return polynomialcf(a, m)
    end
end

function polynomialcf(f, dom::Tuple, m::Int)
    fun = Fun(f, Chebyshev(Interval(check_endpoints(dom)...)))
    return polynomialcf(fun, m)
end

function polynomialcf(a::AbstractVector{T}, m::Int) where {T <: AbstractFloat}
    @assert m <= length(a)
    M = length(a) - 1

    # Trivial case
    if m == M - 1
        return a[1:M], abs(a[M+1])
    end

    c = @views a[m+2:M+1]
    nc = length(c)
    if length(c) >= 32
        D, V = eigs(hankel(c); nev = 1, which = :LM, v0 = ones(T, nc) ./ nc)
        s, u = only(D)::T, vec(V)::Vector{T} # need the asserts since `eigs` is not type-stable
    else
        F = eigen(hankel(c), nc:nc)
        s, u = only(F.values), F.vectors[:, 1]
    end

    u1 = u[1]
    uu = @views u[2:M-m]
    b = [zeros(T, 2m + 1); c]
    for k in -m:m
        b[m-k+1] = @views -dot(b[m-k+2:M-k], uu) / u1
    end
    @views b[m+2:2m+1] .+= b[m:-1:1]
    p = @views a[1:m+1] - b[m+1:2m+1]

    return p, s
end

# Build a Hankel matrix from the given vector of coefficients
function hankel(c::AbstractVector{T}) where {T}
    n = length(c)
    H = zeros(T, n, n)
    for j in 1:n
        for i in 1:n-j+1
            H[i, j] = c[i+j-1]
        end
    end
    return Symmetric(H)
end

# Convert coefficients from the Chebyshev(-1..1) space to Taylor() space, i.e. to the monomial basis
function poly(cheb::AbstractVector{T}) where {T}
    if length(cheb) <= 2
        return copy(cheb)
    end

    n = length(cheb)
    t, tlast1, tlast2 = zeros(T, n), zeros(T, n), zeros(T, n)
    tlast1[2] = 1
    tlast2[1] = 1

    mono = zeros(T, n)
    mono[end] = cheb[2]
    mono[end-1] = cheb[1]

    for k in 3:n
        for j in 1:k
            t[j] = (j == 1 ? zero(T) : 2 * tlast1[j-1]) - (j <= k - 2 ? tlast2[j] : zero(T))
            mono[end-k+j] = cheb[k] * t[j] + (j <= k - 1 ? mono[end-k+j+1] : zero(T))
        end
        tlast2 .= tlast1
        tlast1 .= t
    end

    return mono
end
poly(f::Fun) = poly(coefficients(check_fun(f)))

end # module CaratheodoryFejerApprox
