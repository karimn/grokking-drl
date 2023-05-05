using ReinforcementLearningBase
using Random
using Pipe
using DataFrames, DataFramesMeta
using Gadfly

include("abstract.jl")
include("strategy.jl")

s = εGreedyExpStrategy(0.5, 0.01, 2000)

@pipe DataFrame(t = 1:2000) |>
    @transform!(_, :ε = [decay!(s) for _ in 1:nrow(_)]) |> 
    plot(_, x = :t, y = :ε, Geom.line())
