import Flux, CUDA
import Statistics
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("buffers.jl")
include("drl-algo.jl")
include("fcq.jl")
include("fcduelingq.jl")
include("learners.jl")

env = CartPoleEnv()
bestscore = 0
bestagent = nothing
dqnresults = []

usegpu = true 
numlearners = 5

wmse(ŷ, y, w; agg = Statistics.mean) = agg((w.*(ŷ - y)).^2) 

prog = Progress(numlearners)

@threads for _ in 1:numlearners
    learner = DQNLearner{FCDuelingQ}(env, [512, 128], Flux.RMSProp(0.0007), 1, 10; isdouble = true, tau = 0.9, lossfn = wmse, usegpu)
    buffer = PrioritizedReplayBuffer{50_000, 64}(Float32(0.6), Float32(0.1), Float32(0.99992))
    results, (evalscore, _) = train!(learner, εGreedyExpStrategy(1.0, 0.3, 20_0000), GreedyStrategy(), 1.0, 20, 10_000, buffer; usegpu)

    push!(dqnresults, results)

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)