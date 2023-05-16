struct DoubleNetworkActorCriticModel{PM, VM} <: AbstractActorCriticModel where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel::PM
    valuemodel::VM
end

function DoubleNetworkActorCriticModel{PM, VM}(ninputdims::Int, noutputdims::Int, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser;
                                               usegpu = true) where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel = PM(ninputdims, noutputdims, policyopt; hiddendims = policyhiddendims, usegpu)
    valuemodel = VM(ninputdims, valueopt; hiddendims = valuehiddendims, usegpu)

    DoubleNetworkActorCriticModel{PM, VM}(policymodel, valuemodel)
end

value(m::DoubleNetworkActorCriticModel, state) = m.valuemodel(state)

selectaction(m::DoubleNetworkActorCriticModel, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) = selectaction(m.policymodel, state; rng, usegpu)

train!(m::DoubleNetworkActorCriticModel, states, actions, rewards; β, γ = 1.0, updatemodels = true) = train!(m.policymodel, m.valuemodel, states, actions, rewards; β, γ, updatemodels)
train!(m::DoubleNetworkActorCriticModel, states, actions, rewards, λ; β, γ = 1.0, updatemodels = true) = train!(m.policymodel, m.valuemodel, states, actions, rewards, λ; β, γ, updatemodels)

function update!(m::DoubleNetworkActorCriticModel, grads) 
    prevpolicymodel = m.policymodel
    prevvaluemodel = m.valuemodel

    Flux.update!(opt(m.policymodel), m.policymodel.model, grads[1][1])
    Flux.update!(opt(m.valuemodel), m.valuemodel.model, grads[2][1])

    if any(layerparam -> any(isnan, layerparam), Flux.params(m.policymodel.model)) 
        badpolicymodel = m.policymodel  
        badvaluemodel = m.valuemodel  

        throw(NaNParamException(badpolicymodel, prevpolicymodel, states, actions))
    end
end