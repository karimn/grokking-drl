import Flux, CUDA
import Statistics, Distributions
import Logging
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames
using Dates: now
using Base: put!, take!

include("abstract.jl")
include("util.jl")
include("cartpole.jl")
include("fcq.jl")
include("fcdap.jl")
include("fcv.jl")
include("doublenetworkac.jl")
include("policylearners.jl")
include("a3c.jl")
include("parallelenv.jl")

const usegpu = true 
const numlearners = 5
const maxminutes = 20
const maxepisodes = 10_000
const max_nsteps = 50
const nworkers = 8
const max_cartpole_steps = 500

env = CartPoleEnv(max_steps = max_cartpole_steps, T = Float32)

logger = Logging.SimpleLogger(stdout, Logging.Debug)
oldlogger = Logging.global_logger(logger)

# REINFORCE

bestscore = 0
bestagent = nothing
dqnresults = []

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

# VPG

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

for _ in 1:numlearners
    # This is equivalent to:
    # learner = A3CLearner{DoubleNetworkActorCriticModel{FCDAP, FCV}}(env, [128, 64], [256, 128], Flux.RMSProp(0.0005), Flux.RMSProp(0.0007); max_nsteps = 50_000, nworkers = 1, β = 0.001, usegpu)
    learner = VPGLearner{DoubleNetworkActorCriticModel{FCDAP, FCV}}(env, [128, 64], [256, 128], Flux.RMSProp(0.0005), Flux.RMSProp(0.0007); β = 0.001, usegpu)
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

# A3C

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

for _ in 1:numlearners
    learner = A3CLearner{DoubleNetworkActorCriticModel{FCDAP, FCV}}(env, [128, 64], [256, 128], Flux.RMSProp(0.0005), Flux.RMSProp(0.0007); max_nsteps, nworkers = 8, β = 0.001, usegpu)
    results, (evalscore, _) = train!(learner; maxminutes = 10, maxepisodes, usegpu)

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)

# GAE

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

ex = nothing

try
    for _ in 1:numlearners
        learner = GAELearner(DoubleNetworkActorCriticModel{FCDAP, FCV}, env, [128, 64], [256, 128], Flux.Adam(0.0005), Flux.RMSProp(0.0007); max_nsteps, nworkers, β = 0.001, λ = 0.95, usegpu)
        results, (evalscore, _) = train!(learner; maxminutes = 10, maxepisodes, usegpu)

        push!(dqnresults, results)

        @info "Learning completed." evalscore

        if evalscore >= bestscore
            global bestscore = evalscore
            global bestagent = learner 
        end

        next!(prog)
    end
catch e
    print("Exception!")
    global ex = e
end

finish!(prog)

