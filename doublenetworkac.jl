struct DoubleNetworkActorCriticModel{PM, VM} <: AbstractActorCriticModel where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel::PM
    valuemodel::VM
    opt
end

@functor DoubleNetworkActorCriticModel (policymodel, valuemodel)

function DoubleNetworkActorCriticModel{PM, VM}(ninputdims::Int, noutputdims::Int, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser;
                                               usegpu = true) where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel = PM(ninputdims, noutputdims, policyhiddendims; usegpu)
    valuemodel = VM(ninputdims, valuehiddendims; usegpu)

    DoubleNetworkActorCriticModel{PM, VM}(policymodel, valuemodel, (policymodel = Flux.setup(policyopt, policymodel), valuemodel = Flux.setup(valueopt, valuemodel)))
end

𝒱(m::DoubleNetworkActorCriticModel, state) = 𝒱(m.valuemodel, state)
π(m::DoubleNetworkActorCriticModel, state) = π(m.policymodel, state)
𝒬(m::DoubleNetworkActorCriticModel, state, action) = 𝒬(m.valuemodel, state, action)
ℒᵥ(m::DoubleNetworkActorCriticModel, v̂, v) = ℒ(m.valuemodel, v̂, v) 

selectaction(m::DoubleNetworkActorCriticModel, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) = selectaction(m.policymodel, state; rng, usegpu)
selectgreedyaction(m::DoubleNetworkActorCriticModel, state; usegpu = true) = selectgreedyaction(m.policymodel, state; usegpu)

function Flux.update!(::Nothing, m::DoubleNetworkActorCriticModel, grads::Tuple) 
    Flux.update!(m, grads)
end

function Flux.update!(m::DoubleNetworkActorCriticModel, grads...) 
    Flux.update!.(Ref(m.opt), Ref(m), grads)
end

function Flux.train!(m::DoubleNetworkActorCriticModel, states, actions, rewards, λ::Union{Nothing, Float32} = nothing, opt::Nothing = nothing; valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0, γ = 1.0) 
    flatrewards = reduce(vcat, rewards)
    flatactions = reduce(vcat, actions)

    @assert length(flatrewards) == length(rewards) && length(flatactions) == length(actions) "Only handling a single environment"

    train!(m, states, flatactions, flatrewards, λ; valuelossweight, policylossweight, entropylossweight, γ)
end

update_target_policy_model!(to::DoubleNetworkActorCriticModel, from::DoubleNetworkActorCriticModel; τ = 1.0) = update_target_model!(to.policymodel, from.policymodel)
update_target_value_model!(to::DoubleNetworkActorCriticModel, from::DoubleNetworkActorCriticModel; τ = 1.0) = update_target_model!(to.valuemodel, from.valuemodel)