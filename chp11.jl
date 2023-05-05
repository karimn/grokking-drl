import Flux, CUDA
import Statistics, Distributions
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames

include("util.jl")
include("abstract.jl")
include("cartpole.jl")
include("fcq.jl")
include("fcdap.jl")
include("policylearners.jl")

env = CartPoleEnv()
bestscore = 0
bestagent = nothing
dqnresults = []

usegpu = true 
numlearners = 5

prog = Progress(numlearners)

lk = ReentrantLock()

@threads for _ in 1:numlearners
    learner = REINFORCELearner{FCDAP}(env, [512, 128], Flux.RMSProp(0.0007); usegpu)
    results, (evalscore, _) = train!(learner, 1.0, 20, 10_000; usegpu)

    lock(lk) do
        push!(dqnresults, results)

        if evalscore >= bestscore
            global bestscore = evalscore
            global bestagent = learner 
        end
    end

    next!(prog)
end

finish!(prog)