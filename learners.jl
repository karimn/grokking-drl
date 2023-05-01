struct FQNLearner{E} <: AbstractLearner where {E <: AbstractEnv}
    onlinemodel::FCQ
    epochs::Int
    env::E
end

function FQNLearner(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int; usegpu = true) where {E <: AbstractEnv}
    nS, nA = spacedim(env), nactions(env)
    model = FCQ(nS, nA, opt, hiddendims = hiddendims, usegpu = usegpu)

    return FQNLearner{E}(model, epochs, env)
end

optimizemodel!(learner::FQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, usegpu = usegpu)

selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractLearner = selectaction(trainstrategy, learner.onlinemodel, currstate, rng = rng, usegpu = usegpu)

evaluate(evalstrategy::AbstractStrategy, learner::L, env::E; nepisodes = 1, usegpu = true) where {E <: AbstractEnv, L <: AbstractLearner} = evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)

(learner::FQNLearner)(state) = learner.onlinemodel(state)

save(learner::L, filename) where L <: AbstractLearner = save(learner.onlinemodel, filename)

mutable struct DQNLearner{E} <: AbstractLearner where {E <: AbstractEnv}
    onlinemodel::FCQ
    targetmodel::FCQ
    epochs::Int
    env::E
    updatemodelsteps::Int
    isdoublelearner::Bool
end

function DQNLearner(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int, updatemodelsteps::Int; isdouble = false, usegpu = true) where {E <: AbstractEnv}
    nS, nA = spacedim(env), nactions(env)
    onlinemodel = FCQ(nS, nA, opt; hiddendims, usegpu = usegpu)
    targetmodel = FCQ(nS, nA, opt; hiddendims, usegpu = usegpu)

    return DQNLearner{E}(onlinemodel, targetmodel, epochs, env, updatemodelsteps, isdouble)
end

function optimizemodel!(learner::DQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer 
    optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, argmaxmodel = learner.isdoublelearner ? learner.onlinemodel : learner.targetmodel, targetmodel = learner.targetmodel, usegpu = usegpu)

    if (step % learner.updatemodelsteps) == 0 
        learner.targetmodel = deepcopy(learner.onlinemodel) 
    end
end

(learner::DQNLearner)(state) = learner.onlinemodel(state)