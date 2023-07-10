import Flux, CUDA
import Statistics
import ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
import Logging
using Dates: now
using Random, Base.Threads, Pipe, BSON
using ReinforcementLearningBase: RLBase, AbstractEnv
using ReinforcementLearningEnvironments: RLEnvs, reset!, state, reward, is_terminated
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames

include("util.jl")
include("abstract.jl")
include("env.jl")
include("strategy.jl")
include("buffers.jl")
include("valuemodel.jl")
include("fcq.jl")
include("fcduelingq.jl")
include("valuelearners.jl")

usegpu = true 
numlearners = 5
maxminutes = 20
maxepisodes = 10_000
max_cartpole_steps = 500

# logger = Logging.SimpleLogger(stdout, Logging.Debug)
# oldlogger = Logging.global_logger(logger)

env = CartPoleEnv(max_steps = max_cartpole_steps, T = Float32)
bestscore = 0
bestagent = nothing
dqnresults = []

wmse(ŷ, y, w; agg = Statistics.mean) = agg((w.*(ŷ - y)).^2) 

prog = Progress(numlearners)

lk = ReentrantLock()

for _ in 1:numlearners
    learner = DQNLearner{FCDuelingQ}(env, [512, 128], Flux.RMSProp(0.0001); epochs = 1, updatemodelsteps = 1, τ = 0.1, isdouble = true, lossfn = wmse, usegpu)
    buffer = PrioritizedReplayBuffer(20_000, 64, alpha = 0.6f0, beta = 0.1f0, betarate = 0.99992f0)
    results, (evalscore, _) = train!(learner, εGreedyExpStrategy(ε = 1.0, min_ε = 0.3, decaysteps = 20_000), GreedyStrategy(), buffer; maxminutes, maxepisodes, usegpu)

    lock(lk) do
        @info "Learning completed." evalscore

        push!(dqnresults, results) 

        if evalscore >= bestscore
            global bestscore = evalscore
            global bestagent = learner 
        end
    end

    next!(prog)
end

finish!(prog) 