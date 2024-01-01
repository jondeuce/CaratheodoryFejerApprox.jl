function [p, q, s] = cfapprox(f, dom, m, n)
%CFAPPROX   Caratheodory-Fejer approximation
%   [P, Q, S] = CFAPPROX(F, DOM, M, N) computes a type (M, N) rational CF
%   approximant to fun = CHEBFUN(f, dom) via Chebfun's CF(fun, M, N).
%   Returns vectors P and Q, the Chebyshev coefficients of the numerator
%   and denominator polynomials of the approximant, and S, the associated
%   CF singular value, an approximation to the minimax error.
    if numel(dom) ~= 2
        error('Domain must be a 2-vector.');
    end
    fun = chebfun(f, dom(:)');
    [p, q, ~, s] = cf(fun, m, n);
    p = chebcoeffs(p);
    q = chebcoeffs(q);
end
