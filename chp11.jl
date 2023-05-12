import Flux, CUDA
import Statistics, Distributions
import Logging
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames
using Dates: now

include("util.jl")
include("abstract.jl")
include("cartpole.jl")
include("fcq.jl")
include("fcdap.jl")
include("fcv.jl")
include("policylearners.jl")

usegpu = true 
numlearners = 5
maxminutes = 20
maxepisodes = 10_000
max_cartpole_steps = 500

env = CartPoleEnv(max_steps = max_cartpole_steps, T = Float32)

bestscore = 0
bestagent = nothing
dqnresults = []

# logger = Logging.SimpleLogger(stdout, Logging.Debug)
# oldlogger = Logging.global_logger(logger)

prog = Progress(numlearners)

for _ in 1:numlearners
    learner = REINFORCELearner{FCDAP}(env, [512, 128], Flux.RMSProp(0.0007); usegpu)
    results, (evalscore, _) = train!(learner; maxminutes, maxepisodes, usegpu)

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

for _ in 1:numlearners
    learner = VPGLearner{FCDAP, FCV}(env, [128, 64], [256, 128], Flux.RMSProp(0.0005), Flux.RMSProp(0.0007); β = 0.001, usegpu)
    results, (evalscore, _) = train!(learner; maxminutes, maxepisodes, usegpu)

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)
