using Aqua
using CaratheodoryFejerApprox
using Test

@testset "CaratheodoryFejerApprox.jl" begin
    if get(ENV, "CI", "false") == "false"
        @testset "Chebfun" begin
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
