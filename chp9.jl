import Flux, CUDA
import Statistics
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using StatsBase: sample
using DataFrames

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

usegpu = true 
numlearners = 5

prog = Progress(numlearners)

@threads for _ in 1:numlearners
    learner = DQNLearner{FCQ}(env, [512, 128], Flux.RMSProp(0.0005), 1, 10; isdouble = true, usegpu)
    buffer = ReplayBuffer{50_000, 64}()
    results, (evalscore, _) = train!(learner, εGreedyStrategy(0.5), GreedyStrategy(), 1.0, 20, 10_000, buffer; usegpu)
    push!(dqnresults, results)

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)