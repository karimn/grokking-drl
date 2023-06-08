import Flux, CUDA, Functors
import Statistics, Distributions
import Logging, LoggingExtras
import JLD2
import PyCall
import DomainSets
using Functors: @functor
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using Statistics: mean, std
using StatsBase: sample, wsample
using DataFrames
using Dates: now
using Logging: @logmsg

include("abstract.jl")
include("util.jl")
include("env.jl")
include("buffers.jl")
include("fcqv.jl")
include("fcqsa.jl")
include("fctqv.jl")
include("policymodel.jl")
include("doublenetworkac.jl")
include("fcdp.jl")
include("strategy.jl")
include("valuelearners.jl")
include("ddpg.jl")
include("td3.jl")

const numlearners = 5
const usegpu = true 
const γ = 0.99f0 

env = PendulumEnv((-1, 1), max_steps = 200, T = Float32)
hopperenv = HopperEnv{Float32}()

#logger = Logging.ConsoleLogger(stdout, Logging.Debug)
#oldlogger = Logging.global_logger(logger)
#Logging.shouldlog(::Logging.ConsoleLogger, level, _module::Module, group, id) = nameof(_module) == :Main
#filelogger = LoggingExtras.MinLevelLogger(LoggingExtras.FileLogger("td3.txt"), Logging.LogLevel(-1001))
#Logging.shouldlog(::LoggingExtras.FileLogger, level, _module::Module, group, id) = nameof(_module) != :ChainRules

#Logging.with_logger(filelogger) do
    # learner = TD3Learner(hopperenv, ReplayBuffer(1_000_000, 256), [256, 256], [256, 256], Flux.Adam(0.0001), Flux.Adam(0.0001); 
    #                      update_value_model_steps = 2, update_policy_model_steps = 2, train_policy_model_steps = 2, τ = 0.01, policy_noise_ratio = 0.1f0, policy_noise_clip_ratio = 0.5f0, usegpu) 
    # results, (evalscore, _) = train!(learner, NormalNoiseDecayStrategy(decaysteps = 200_000), ContinuousGreedyStrategy(); γ, maxminutes = 300, maxepisodes = 10_000, goal_mean_reward = 1500, usegpu) 

    # @info "Learning completed." evalscore
#end 

# DDPG

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

ex = nothing

for _ in 1:numlearners
    learner = DDPGLearner(env, ReplayBuffer(100_000, 256), [256, 256], [256, 256], Flux.Adam(0.0003), Flux.Adam(0.0003); updatemodelsteps = 1, τ = 0.005, usegpu) 
    results, (evalscore, _) = train!(learner, NormalNoiseGreedyStrategy(), ContinuousGreedyStrategy(); γ, maxminutes = 20, maxepisodes = 500, goal_mean_reward = -150.0, usegpu) 

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)

# TD3 

bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

ex = nothing

for _ in 1:numlearners
    learner = TD3Learner(hopperenv, ReplayBuffer(1_000_000, 256), [256, 256], [256, 256], Flux.Adam(0.0003), Flux.Adam(0.0003); 
                         update_value_model_steps = 2, update_policy_model_steps = 2, train_policy_model_steps = 2, τ = 0.01, policy_noise_ratio = 0.1f0, policy_noise_clip_ratio = 0.5f0, usegpu) 
    results, (evalscore, _) = train!(learner, NormalNoiseDecayStrategy(decaysteps = 200_000), ContinuousGreedyStrategy(); γ, maxminutes = 300, maxepisodes = 10_000, goal_mean_reward = 1500, usegpu) 

    push!(dqnresults, results)

    @info "Learning completed." evalscore

    if evalscore >= bestscore
        global bestscore = evalscore
        global bestagent = learner 
    end

    next!(prog)
end

finish!(prog)