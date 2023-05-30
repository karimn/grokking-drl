import Flux, CUDA, Functors
import Statistics, Distributions
import Logging
import JLD2
using Functors: @functor
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using ProgressMeter
using Statistics: mean, std
using StatsBase: sample, wsample
using DataFrames
using Dates: now

include("abstract.jl")
include("util.jl")
include("env.jl")
include("buffers.jl")
include("fcqv.jl")
include("policymodel.jl")
include("doublenetworkac.jl")
include("fcdp.jl")
include("strategy.jl")
include("valuelearners.jl")
include("ddpg.jl")

const usegpu = true 
const maxminutes = 20
const maxepisodes = 500 
const goal_mean_reward = -150.0 

env = PendulumEnv((-1, 1), max_steps = 200, T = Float32)

logger = Logging.SimpleLogger(stdout, Logging.Debug)
oldlogger = Logging.global_logger(logger)

learner = DDPGLearner(env, [256, 256], [256, 256], Flux.Adam(0.0003), Flux.Adam(0.0003); updatemodelsteps = 1, τ = 0.005, usegpu) 
buffer = ReplayBuffer{100_000, 256}()
results, (evalscore, _) = train!(learner, NormalNoiseGreedyStrategy(), ContinuousGreedyStrategy(), buffer; maxminutes, maxepisodes, goal_mean_reward, usegpu) 