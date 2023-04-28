import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe
using ProgressMeter
using StatsBase: sample
using DataFrames

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("buffers.jl")
include("drl-algo.jl")
include("fcq.jl")
include("nfq.jl")
include("dqn.jl")

env = CartPoleEnv()
bestscore = 0
bestagent = nothing
dqnresults = []

@showprogress for _ in 1:1
    agent = DQN([512, 128], Flux.RMSProp(0.0005), 40, 10)
    #results, (evalscore, _) = train!(agent, env, εGreedyStrategy(0.5), GreedyStrategy(), 1.0, 20, 10_000, Buffer{1024}, usegpu = false)
    results, (evalscore, _) = train!(agent, env, εGreedyStrategy(0.5), GreedyStrategy(), 1.0, 20, 10_000, ReplayBuffer{50_000, 64}, usegpu = true)
    push!(dqnresults, results)

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = agent
    end
end