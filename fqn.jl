struct FQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueModel}
    onlinemodel::M
    epochs::Int
    env::E
end

function FQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; epochs::Int, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true) where {E <: AbstractEnv, M <: AbstractValueModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return FQNLearner{E, M}(model, epochs, env)w
end

optimizemodel!(learner::FQNLearner, experiences::B, γ, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(learner.onlinemodel, experiences, learner.epochs, γ, usegpu = usegpu)

(learner::FQNLearner)(state) = learner.onlinemodel(state)
