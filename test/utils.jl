using CaratheodoryFejerApprox

using ApproxFun: Chebyshev, Fun, Interval
using CaratheodoryFejerApprox: check_endpoints, lazychopcoeffs, parity, normalize_rational
using Statistics: mean

rand_uniform(a::T, b::T) where {T} = a + (b - a) * rand(T)
rand_uniform(a::T, b::T, n::Int) where {T} = a .+ (b - a) .* rand(T, n)
rand_uniform(dom::Tuple = (-1, 1), args...) = rand_uniform(float.(dom)..., args...)

cheb_interval(dom) = cheb_interval(check_endpoints(dom))
cheb_interval(dom::NTuple{2, T}) where {T <: AbstractFloat} = cheb_interval(T, check_endpoints(dom))
cheb_interval(::Type{T}, dom) where {T} = Chebyshev(Interval(float(T).(check_endpoints(dom))...))

build_fun(f::Base.Callable, dom::Tuple = (-1, 1)) = Fun(f, cheb_interval(dom))
build_fun(a::AbstractArray, dom::Tuple = (-1, 1)) = Fun(cheb_interval(float(eltype(a)), dom), float(a))
build_fun((p, q)::NTuple{2, <:AbstractArray}, dom::Tuple = (-1, 1)) = build_fun.(normalize_rational(p, q), (dom,))
build_fun(dom::Tuple = (-1, 1)) = a -> build_fun(a, dom)

function compare_chebcoeffs(a1::AbstractVector{T}, a2::AbstractVector{T}; atol, rtol, parity = :generic) where {T <: AbstractFloat}
    # Compare Chebyshev coefficients of two polynomials
    c1, c2 = lazychopcoeffs(a1; parity), lazychopcoeffs(a2; parity)
    length(c1) == length(c2) || return false
    @views if parity === :even
        pass1 = length(c1) <= 1 || maximum(abs, c1[2:2:end]) <= rtol * sum(abs, c1)
        pass2 = length(c2) <= 1 || maximum(abs, c2[2:2:end]) <= rtol * sum(abs, c2)
        pass3 = isapprox(c1[1:2:end], c2[1:2:end]; atol, rtol)
        pass = pass1 && pass2 && pass3
    elseif parity === :odd
        pass1 = length(c1) <= 1 || maximum(abs, c1[1:2:end]) <= rtol * sum(abs, c1)
        pass2 = length(c2) <= 1 || maximum(abs, c2[1:2:end]) <= rtol * sum(abs, c2)
        pass3 = isapprox(c1[2:2:end], c2[2:2:end]; atol, rtol)
        pass = pass1 && pass2 && pass3
    else
        pass = isapprox(c1, c2; atol, rtol)
    end
    return pass
end

function rand_chebcoeffs(::Type{T} = Float64; n::Int = 100, min = eps(T), even = nothing) where {T <: AbstractFloat}
    # Random exponentially decaying coefficients such that |c[n]| ~ eps(T)
    c = randn(T, n) .* exp.(range(0, T(log(min)); length = n))
    (even isa Bool) && (c[1+even:2:end] .= 0)
    return c
end

function rand_chebfun(dom::NTuple{2, T} = (-1.0, 1.0); kwargs...) where {T <: AbstractFloat}
    # Chebfun with random exponentially decaying coefficients such that |c[n]| ~ eps(T)
    c = rand_chebcoeffs(T; kwargs...)
    return build_fun(c, dom)
end
