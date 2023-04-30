import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe
using ProgressMeter
using StatsBase: sample
using DataFrames
using BSON

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("buffers.jl")
include("drl-algo.jl")
include("fcq.jl")
include("learners.jl")

env = CartPoleEnv()
bestscore = 0
bestagent = nothing
dqnresults = []

@showprogress for _ in 1:5
    learner = DQNLearner(env, [512, 128], Flux.RMSProp(0.0005), 40, 10, usegpu = true)
    results, (evalscore, _) = train!(learner, εGreedyStrategy(0.5), GreedyStrategy(), 1.0, 20, 10_000, ReplayBuffer{50_000, 64}, usegpu = true)
    push!(dqnresults, results)

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = agent
    end
end