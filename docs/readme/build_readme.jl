using Pkg
base_dir = realpath(joinpath(@__DIR__, "../.."))

Pkg.activate(@__DIR__)
Pkg.develop(; path = base_dir)
# Pkg.update()

using Literate

readme_src = joinpath(@__DIR__, "README.jl")
readme_dst = base_dir
Literate.markdown(readme_src, readme_dst; flavor = Literate.CommonMarkFlavor(), execute = true)
