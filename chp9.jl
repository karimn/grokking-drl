
import Flux, CUDA
import Statistics
using Random
using ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
using Pipe

include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("fcq.jl")
include("nfq.jl")
include("dqn.jl")

env = CartPoleEnv()

agent = DQN([512, 128], Flux.RMSProp(0.0005), εGreedyStrategy(0.5), GreedyStrategy(), 1024, 40, 10)

train!(agent, env, 1.0, 20, 10_000, usegpu = true)