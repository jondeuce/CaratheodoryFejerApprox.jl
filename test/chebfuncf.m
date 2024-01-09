function [p, q, s] = chebfuncf(f, varargin)
%CHEBFUNCF  Caratheodory-Fejer approximation
%   [P, Q, S] = CHEBFUNCF(F, DOM, M, N, PARITY) computes a type (M, N) rational CF
%   approximant to fun = CHEBFUN(f, dom) via Chebfun's CF(fun, M, N).
%   Returns vectors P and Q, the Chebyshev coefficients of the numerator
%   and denominator polynomials of the approximant, and S, the associated
%   CF singular value, an approximation to the minimax error.

    [fun, m, n] = parse_inputs(f, varargin{:});
    [p, q, ~, s] = cf(fun, m, n);
    p = chebcoeffs(p);
    q = chebcoeffs(q);
end

function [fun, m, n] = parse_inputs(f, varargin)

    m = [];
    n = [];
    dom = [];
    parity = '';

    for ii = numel(varargin):-1:1
        arg = varargin{ii};
        if ischar(arg)
            if isempty(parity)
                parity = arg;
            else
                error('CHEBFUNCF:invalidInput', 'Multiple parity input arguments supplied.');
            end
        elseif isnumeric(arg)
            if numel(arg) == 1
                if isempty(n)
                    n = arg;
                elseif isempty(m)
                    m = arg;
                else
                    error('CHEBFUNCF:invalidInput', 'More than two approximant degrees specified.');
                end
            elseif numel(arg) == 2
                if isempty(dom)
                    dom = arg;
                else
                    error('CHEBFUNCF:invalidInput', 'Multiple domain input arguments supplied.');
                end
            end
        else
            error('CHEBFUNCF:invalidInput', 'Invalid input arguments.');
        end
    end

    if isempty(m) && isempty(n)
        error('CHEBFUNCF:invalidInput', 'Approximant degrees not specified.');
    elseif isempty(m)
        m = n;
        n = 0;
    end

    if isempty(dom)
        dom = [-1, 1];
    end

    if isa(f, 'chebfun')
        fun = f;
        dom = domain(fun);
    else
        fun = chebfun(f, dom(:)');
    end

    if isempty(parity)
        c = chebcoeffs(fun);
        v = vscale(fun);
        if max(abs(c(2:2:end))) <= eps * v % f is even
            parity = 'even';
        elseif max(abs(c(1:2:end))) <= eps * v % f is odd
            parity = 'odd';
        else
            parity = 'generic';
        end
    end

    % Note: there is a bug in chebfun/cf.m for even/odd symmetric functions when n = 1,
    %       but in this case we know that the approximant is a polynomial.
    if n == 1 && (strcmpi(parity, 'even') || strcmpi(parity, 'odd'))
        n = 0;
    end
end
