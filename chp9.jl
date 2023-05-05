import Flux, CUDA
import Statistics
import ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments
import Logging
using Dates: now
using ReinforcementLearningBase: RLBase, AbstractEnv
using ReinforcementLearningEnvironments: RLEnvs, reset!, state, reward, is_terminated
using Random, Base.Threads, Pipe, BSON
using ProgressMeter
using StatsBase: sample, wsample
using DataFrames

include("util.jl")
include("abstract.jl")
include("cartpole.jl")
include("strategy.jl")
include("buffers.jl")
include("fcq.jl")
include("valuelearners.jl")

#logger = Logging.SimpleLogger(stdout, Logging.Debug)
#oldlogger = Logging.global_logger(logger)

usegpu = true 
numlearners = 5
max_cartpole_steps = 500
maxepisodes = 50_000
maxminutes = 20

env = CartPoleEnv(max_steps = max_cartpole_steps)
bestscore = 0
bestagent = nothing
dqnresults = []

prog = Progress(numlearners)

wmse(ŷ, y, w; agg = Statistics.mean) = agg((w.*(ŷ - y)).^2) 

lk = ReentrantLock()

@threads for _ in 1:numlearners
    learner = DQNLearner{FCQ}(env, [512, 128], Flux.RMSProp(0.0005); epochs = 1, updatemodelsteps = 10, isdouble = true, usegpu)
    buffer = ReplayBuffer{50_000, 64}()
    results, (evalscore, _) = train!(learner, εGreedyStrategy(0.5), GreedyStrategy(), buffer; γ = 1.0, maxminutes, maxepisodes, usegpu)

    lock(lk) do
        push!(dqnresults, results)

        if evalscore >= bestscore
            global bestscore = evalscore
            global bestagent = learner 
        end
    end

    next!(prog)
end

finish!(prog)