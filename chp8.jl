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
include("valuelearners.jl")

env = CartPoleEnv()

learner = FQNLearner{FCQ}(env, [512, 128], Flux.RMSProp(0.0005); epochs = 40, usegpu = true)

buffer = Buffer{1024}()

train!(learner, εGreedyStrategy(0.5), GreedyStrategy(), buffer; gamma = 1.0, maxminutes = 20, maxepisodes = 10_000, usegpu = true)