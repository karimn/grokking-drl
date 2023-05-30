mutable struct DQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueModel}
    onlinemodel::M
    targetmodel::M
    epochs::Int
    env::E
    updatemodelsteps::Int
    isdoublelearner::Bool
    τ::Float32
end

function DQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; 
                       epochs::Int, updatemodelsteps::Int, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), isdouble = false, τ = 1.0, usegpu = true) where {E <: AbstractEnv, M <: AbstractValueModel}

    nS, nA = spacedim(env), nactions(env)
    onlinemodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)
    targetmodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return DQNLearner{E, M}(onlinemodel, targetmodel, epochs, env, updatemodelsteps, isdouble, τ)
end

function update_target_model!(l::DQNLearner) 
    if l.τ < 1.0
        update_target_model!(l.targetmodel, l.onlinemodel, τ = l.τ)
    else
        l.targetmodel = deepcopy(l.onlinemodel)
    end
end

function optimizemodel!(learner::DQNLearner, experiences::B, γ, step; usegpu = true) where B <: AbstractBuffer 
    optimizemodel!(learner.onlinemodel, experiences, learner.epochs, γ, argmaxmodel = learner.isdoublelearner ? learner.onlinemodel : learner.targetmodel, targetmodel = learner.targetmodel, usegpu = usegpu)

    (step % learner.updatemodelsteps) == 0 && update_target_model!(learner)
end

(learner::DQNLearner)(state) = learner.onlinemodel(state)