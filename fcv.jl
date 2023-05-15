struct FCV <: AbstractValueModel
    model
    opt
    lossfn
end

function FCV(inputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return CreateSimpleFCModel(FCV, inputdim, 1, valueopt; hiddendims, actfn, lossfn, usegpu)
end

(m::FCV)(state) = m.model(state) 

function train!(policymodel::AbstractPolicyModel, valuemodel::AbstractValueModel, states, actions, rewards; β, γ = 1.0, updatemodels = true)
    T = length(rewards) 
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 
    pop!(discounts)
    pop!(returns)

    values = nothing 

    vval, vgrads = Flux.withgradient(valuemodel.model, states, returns) do modelchain, s, r
        values = modelchain(s) |> Flux.cpu |> vec 

        return valuemodel.lossfn(values, r)
    end

    value_errors = returns - values

    pval, pgrads = Flux.withgradient((args...) -> policyloss(args...; β), policymodel.model, states, actions, discounts, value_errors) 

    isfinite(vval) || @warn "Value loss is $vval"
    updatemodels && Flux.update!(opt(valuemodel), valuemodel.model, vgrads[1])

    isfinite(pval) || @warn "Policy loss is $pval"
    updatemodels && Flux.update!(opt(policymodel), policymodel.model, pgrads[1])

    return pgrads, vgrads
end

function train!(policymodel::AbstractPolicyModel, valuemodel::AbstractValueModel, states, actions, rewards, λ; β, γ = 1.0, updatemodels = true)
    T = length(rewards) 
    discounts = γ.^range(0, T - 1)
    λ_discounts = (λ * γ).^range(0, T - 1) 
    returns = [sum(discounts[1:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    values = nothing 

    vval, vgrads = Flux.withgradient(valuemodel.model, states, returns) do modelchain, s, r
        values = modelchain(s) |> Flux.cpu |> vec 

        return valuemodel.lossfn(values, r[1:(end - 1)])
    end

    push!(values, last(rewards))

    advs = rewards[1:(end - 1)] + γ * values[2:end] - values[1:(end - 1)]
    gaes = [sum(λ_discounts[1:(T - t)] .* advs[t:end]) for t in 1:(T - 1)] 

    pop!(discounts)

    pval, pgrads = Flux.withgradient((args...) -> policyloss(args...; β), policymodel.model, states, actions, discounts, gaes) 

    #@debug "GAEs" gaes
    #@debug "policy" policy = policymodel(states) 

    isfinite(vval) || @warn "Value loss is $vval"
    updatemodels && Flux.update!(opt(valuemodel), valuemodel.model, vgrads[1])

    isfinite(pval) || @warn "Policy loss is $pval"
    prevpolicymodel = deepcopy(policymodel)
    updatemodels && Flux.update!(opt(policymodel), policymodel.model, pgrads[1])

    # if any(layerparam -> any(isnan, layerparam), Flux.params(policymodel.model)) 
    #     throw(NaNParamException(policymodel, prevpolicymodel, gaes, states, actions, discounts, values, returns, β))
    # end

    return pgrads, vgrads
end