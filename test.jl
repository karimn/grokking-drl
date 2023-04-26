import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("nfq.jl")

env = CartPoleEnv()

agent = NFQ{FCQ}([512, 128], Flux.RMSProp(0.0005), εGreedyStrategy(0.5), GreedyStrategy(), 1024, 40)

train!(agent, env, 1.0, 20, 10_000, usegpu = true)