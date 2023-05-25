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

function optimizemodel!(learner::VPGLearner, states, actions, rewards; usegpu = true) 
    statesdata = @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _)

    laststate = state(learner.env)
    failure = is_terminated(learner.env) && !istruncated(learner.env)

    nextvalue = 𝒱(learner.model, usegpu ? Flux.gpu(laststate) : laststate) |> Flux.cpu |> first
    push!(rewards, failure ? 0.0 : nextvalue)
        
    grads = Flux.withgradient(learner.model, statesdata, actions, rewards; γ = learner.γ, entropylossweight = learner.β)

    try
        Flux.update!(learner.model, grads[1])
    catch e
        throw(GradientException(learner.model, statesdata, actions, nothing, e, nothing, 1, length(rewards), nothing, grads))
    end

    return learner.model   
end