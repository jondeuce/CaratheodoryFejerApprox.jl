fs = {@(x) exp(x), @(x) 1./(1 + 25*x.^2), @(x) exp(-x.^2), @(x) x .* exp(-x.^2), @(x) sinc(10*x)};
parities = {'generic', 'even', 'even', 'odd', 'odd'};
doms = {[-1, 1], [-1, 2]};
dmax = 12;

for i = 1:length(fs)
    for j = 1:length(doms)
        f = fs{i};
        dom = doms{j};
        parity = parities{i};
        fun = chebfun(f, dom);
        fprintf('\nBenchmarking cf(f, m, n) with f=%s on [%d, %d] up to total degree %d\n', func2str(f), dom(1), dom(2), dmax)
        t = timeit(@() benchmark(fun, dmax, parity));
        fprintf('Time: %f s\n', t);
    end
end

function [out] = benchmark(fun, dmax, parity)
    out = 0.0;
    for d = dmax:-1:0
        for m = d:-1:0
            n = d - m;
            [p, q, s] = chebfuncf(fun, m, n);
            out = out + abs(s); % ensure loop body not optimized away
        end
    end
end
