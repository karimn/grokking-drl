import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe
using StatsBase: sample
using DataFrames
using BSON

include("cartpole.jl")
include("abstract.jl")
include("strategy.jl")
include("buffers.jl")
include("fcq.jl")
include("learners.jl")
include("drl-algo.jl")

env = CartPoleEnv()

learner = FQNLearner{FCQ}(env, [512, 128], Flux.RMSProp(0.0005), 40, usegpu = true)

train!(learner, εGreedyStrategy(0.5), GreedyStrategy(), 1.0, 20, 10_000, Buffer{1024}, usegpu = true)