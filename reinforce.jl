struct REINFORCELearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractPolicyModel}
    policymodel::M
    policyopt
    epochs::Int
    env::E
    γ::Float32 
end

function REINFORCELearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractPolicyModel}
    nS, nA = spacedim(env), nactions(env)
    policymodel = M(nS, nA; hiddendims, usegpu)

    return REINFORCELearner{E, M}(policymodel, Flux.setup(opt, policymodel), epochs, env, γ)
end

policymodel(m::REINFORCELearner) = m.policymodel
environment(m::REINFORCELearner) = m.env
opt(m::REINFORCELearner) = m.policyopt