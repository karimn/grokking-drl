struct VPGLearner{E, M} <: AbstractActorCriticLearner where {E <: AbstractEnv}
    model::DoubleNetworkActorCriticModel
    epochs::Int
    env::E
    γ::Float32
    β::Float32
end

function VPGLearner{DoubleNetworkActorCriticModel{PM, VM}}(env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                       β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, PM, VM}
    nS, nA = spacedim(env), nactions(env)
    model = DoubleNetworkActorCriticModel{PM, VM}(nS, nA, policyhiddendims, valuehiddendims, policyopt, valueopt; usegpu)

    return VPGLearner{E, DoubleNetworkActorCriticModel{PM, VM}}(model, epochs, env, γ, β)
end

environment(m::VPGLearner) = m.env
model(m::VPGLearner) = m.model
policymodel(m::VPGLearner) = model(m)
discount(m::VPGLearner) = m.γ 
entropylossweight(m::VPGLearner) = m.β
