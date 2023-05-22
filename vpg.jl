struct VPGLearner{E, M} <: AbstractActorCriticLearner where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    model::M
    epochs::Int
    env::E
    γ::Float32
    β::Float32
end

function VPGLearner{M}(env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                       β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, policyhiddendims, valuehiddendims, policyopt, valueopt; usegpu)

    return VPGLearner{E, M}(model, epochs, env, γ, β)
end

environment(m::VPGLearner) = m.env
model(m::VPGLearner) = m.model
policymodel(m::VPGLearner) = model(m)
discount(m::VPGLearner) = m.γ 
entropylossweight(m::VPGLearner) = m.β
