using Pkg
base_dir = realpath(joinpath(@__DIR__, "../../.."))

Pkg.activate(@__DIR__)
Pkg.develop(; path = base_dir)
# Pkg.update()

using CaratheodoryFejerApprox
using Literate

function replify(content)
    # Combine Julia output block with the previous code block
    content = replace(content, "````\n\n````\n" => "")
    lines = split(content, '\n')

    # For lines ending in "#repl", add "julia> " to the beginning of the line
    for (i, line) in enumerate(lines)
        if endswith(line, "#repl")
            lines[i] = "julia> " * split(line, "#repl")[1]
        end
    end

    # Find blocks whose first line ends with "#repl-block-start" and last line ends with "#repl-block-end"
    Istart = findall(endswith("#repl-block-start"), lines)
    Iend = findall(endswith("#repl-block-end"), lines)
    @assert length(Istart) == length(Iend) && all(Istart .< Iend) "Mismatched repl-blocks"

    # For each block, add "julia> " to the beginning of the first line and indent the rest
    for (istart, iend) in zip(Istart, Iend)
        lines[istart] = "julia> " * split(lines[istart], "#repl-block-start")[1]
        for i in (istart+1):iend
            lines[i] = "       " * lines[i]
        end
        lines[iend] = split(lines[iend], "#repl-block-end")[1]
    end

    content = join(chomp.(lines), '\n')
    return content
end

readme_src = joinpath(@__DIR__, "README.jl")
readme_dst = base_dir
Literate.markdown(readme_src, readme_dst; flavor = Literate.CommonMarkFlavor(), postprocess = replify, execute = true)
