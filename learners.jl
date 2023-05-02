struct FQNLearner{E, M} <: AbstractLearner where {E <: AbstractEnv, M <: AbstractModel}
    onlinemodel::M
    epochs::Int
    env::E
end

function FQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int; lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), usegpu = true) where {E <: AbstractEnv, M <: AbstractModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return FQNLearner{E, M}(model, epochs, env)
end

optimizemodel!(learner::FQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, usegpu = usegpu)

selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractLearner = selectaction(trainstrategy, learner.onlinemodel, currstate, rng = rng, usegpu = usegpu)

evaluate(evalstrategy::AbstractStrategy, learner::L, env::E; nepisodes = 1, usegpu = true) where {E <: AbstractEnv, L <: AbstractLearner} = evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)

(learner::FQNLearner)(state) = learner.onlinemodel(state)

save(learner::L, filename) where L <: AbstractLearner = save(learner.onlinemodel, filename)

mutable struct DQNLearner{E, M} <: AbstractLearner where {E <: AbstractEnv, M <: AbstractModel}
    onlinemodel::M
    targetmodel::M
    epochs::Int
    env::E
    updatemodelsteps::Int
    isdoublelearner::Bool
    tau::Float32
end

function DQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int, updatemodelsteps::Int; 
                       lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), isdouble = false, tau = 0.0, usegpu = true) where {E <: AbstractEnv, M <: AbstractModel}

    nS, nA = spacedim(env), nactions(env)
    onlinemodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)
    targetmodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return DQNLearner{E, M}(onlinemodel, targetmodel, epochs, env, updatemodelsteps, isdouble, tau)
end

function optimizemodel!(learner::DQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer 
    optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, argmaxmodel = learner.isdoublelearner ? learner.onlinemodel : learner.targetmodel, targetmodel = learner.targetmodel, usegpu = usegpu)

    if learner.tau > 0
        newparams = [(1 - learner.tau) * targetp + learner.tau * onlinep for (targetp, onlinep) in zip(Flux.params(learner.targetmodel), Flux.params(learner.onlinemodel))]
        Flux.loadparams!(learner.targetmodel, newparams)
    elseif (step % learner.updatemodelsteps) == 0 
        learner.targetmodel = deepcopy(learner.onlinemodel) 
    end
end

(learner::DQNLearner)(state) = learner.onlinemodel(state)