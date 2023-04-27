
import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe
using ProgressMeter

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("drl-algo.jl")
include("fcq.jl")
include("nfq.jl")
include("dqn.jl")

env = CartPoleEnv()
bestscore = 0
bestagent = nothing
dqnresults = []

@showprogress for _ in 1:5
    agent = DQN([512, 128], Flux.RMSProp(0.0005), εGreedyStrategy(0.5), GreedyStrategy(), 1024, 40, 10)
    results, (evalscore, _) = train!(agent, env, 1.0, 20, 10_000, usegpu = true)
    push!(dqnresults, results)

    if evalscore > bestscore
        global bestscore = evalscore
        global bestagent = agent
    end
end