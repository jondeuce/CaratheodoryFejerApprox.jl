using Aqua
using CaratheodoryFejerApprox
using Test

include("utils.jl")

@testset "CaratheodoryFejerApprox.jl" begin
    @testset "Julia Tests" verbose = true begin
        include("julia_tests.jl")
    end

    if get(ENV, "CI", "false") == "false"
        @testset "Matlab Tests" verbose = true begin
            # Only run Chebfun tests if testing locally
            include("chebfun.jl")
        end
    end

    @testset "Code quality (Aqua.jl)" begin
        # Typically causes a lot of false positives with ambiguities and/or unbound args checks;
        # unfortunately have to periodically check this manually
        Aqua.test_all(CaratheodoryFejerApprox; ambiguities = false, unbound_args = true)
    end
end
