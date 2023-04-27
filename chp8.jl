import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe
using StatsBase: sample
using DataFrames

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("buffers.jl")
include("drl-algo.jl")
include("fcq.jl")
include("nfq.jl")

env = CartPoleEnv()

agent = NFQ([512, 128], Flux.RMSProp(0.0005), εGreedyStrategy(0.5), GreedyStrategy(), 40)

train!(agent, env, 1.0, 20, 10_000, Buffer{1024}, usegpu = true)