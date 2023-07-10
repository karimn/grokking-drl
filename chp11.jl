import Flux, CUDA, Functors
import Statistics, Distributions
import Logging
import JLD2
import PyCall
using Functors: @functor
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using Statistics: mean, std
using StatsBase: sample, wsample
using DataFrames
using Dates: now
using Base: put!, take!

include("abstract.jl")
include("util.jl")
include("env.jl")
include("parallelenv.jl")
include("policymodel.jl")
include("valuemodel.jl")
include("fcq.jl")
include("fcdap.jl")
include("fcv.jl")
include("fcac.jl")
include("doublenetworkac.jl")
include("vpg.jl")
include("a3c.jl")
include("policylearners.jl")
include("reinforce.jl")
include("a2c.jl")

const usegpu = true 
const numlearners = 5
const maxminutes = 20
const maxepisodes = 10_000
const max_nsteps = 50
const nworkers = 8
const max_cartpole_steps = 500
const goal_mean_reward = 475
const γ = 0.99f0

env = CartPoleEnv(max_steps = max_cartpole_steps, T = Float32)
parenv = ParallelEnv(env, nworkers)

#io = open("log.txt", "w")
logger = Logging.SimpleLogger(stdout, Logging.Debug)
#logger = Logging.SimpleLogger(io, Logging.Debug)
oldlogger = Logging.global_logger(logger)
Logging.shouldlog(::Logging.ConsoleLogger, level, _module::Module, group, id) = nameof(_module) == :Main

# Logging.with_logger(logger) do
# end

# REINFORCE

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

for _ in 1:numlearners
    learner = REINFORCELearner{FCDAP}(env, [512, 128], Flux.RMSProp(0.0007); usegpu)
    results, (evalscore, _) = train!(learner; maxminutes, maxepisodes, goal_mean_reward, usegpu)

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
    learner = VPGLearner{DoubleNetworkActorCriticModel{FCDAP, FCV}}(env, [128, 64], [256, 128], Flux.RMSProp(0.0005), Flux.RMSProp(0.0007); β = 0.001, usegpu)
    results, (evalscore, _) = train!(learner; maxminutes, maxepisodes, goal_mean_reward, usegpu)

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
    results, (evalscore, _) = train!(learner; maxminutes = 10, maxepisodes, goal_mean_reward, usegpu)

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

for _ in 1:numlearners
    # The policy learning rate here is lower than in the book's notebook. I found that if it is 0.0005 I get degeneracy which leads to NaN gradients. I haven't tried other values yet.
    learner = GAELearner(DoubleNetworkActorCriticModel{FCDAP, FCV}, env, [128, 64], [256, 128], Flux.Optimiser(Flux.ClipNorm(1), Flux.Adam(0.00005)), Flux.RMSProp(0.0007); 
                            max_nsteps, nworkers, γ, β = 0.001, λ = 0.95, usegpu)
    results, (evalscore, _) = train!(learner; maxminutes = 10, goal_mean_reward, maxepisodes, usegpu)

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)

# A2C 

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

ex = nothing

for _ in 1:numlearners
    # Lower learning rate here too to avoid degeneracy.
    learner = A2CLearner{FCAC}(parenv, [256, 128], Flux.Optimiser(Flux.ClipNorm(1.0), Flux.RMSProp(0.001)); 
                                max_nsteps = 10, nworkers, γ, λ = 0.95, policylossweight = 1.0, valuelossweight = 0.6, entropylossweight = 0.001, usegpu)

    results, (evalscore, _) = train!(learner; maxminutes = 10, maxepisodes, goal_mean_reward, usegpu)

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)
