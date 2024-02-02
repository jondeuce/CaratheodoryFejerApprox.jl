using CaratheodoryFejerApprox
using Documenter

DocMeta.setdocmeta!(CaratheodoryFejerApprox, :DocTestSetup, :(using CaratheodoryFejerApprox); recursive = true)

makedocs(;
    modules = [CaratheodoryFejerApprox],
    authors = "Jonathan Doucette <jdoucette@physics.ubc.ca> and contributors",
    repo = "https://github.com/jondeuce/CaratheodoryFejerApprox.jl/blob/{commit}{path}#{line}",
    sitename = "CaratheodoryFejerApprox.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jondeuce.github.io/CaratheodoryFejerApprox.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/jondeuce/CaratheodoryFejerApprox.jl",
    devbranch = "master",
)
