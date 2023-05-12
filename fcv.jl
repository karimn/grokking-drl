struct FCV <: AbstractValueModel
    model
    opt
    lossfn
end

function FCV(inputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return CreateSimpleFCModel(FCV, inputdim, 1, valueopt; hiddendims, actfn, lossfn, usegpu)
end

(m::FCV)(state) = m.model(state) 

function train!(policymodel::PM, valuemodel::VM, states, actions, rewards; β, γ = 1.0) where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    T = length(rewards) 
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 
    discounts = discounts[1:(end - 1)] 
    returns = returns[1:(end - 1)]

    value = nothing 

    vval, vgrads = Flux.withgradient(valuemodel.model, states, returns) do modelchain, s, r
        value = modelchain(s) |> Flux.cpu |> vec 

        return valuemodel.lossfn(r, value)
    end

    value_error = returns - value

    pval, pgrads = Flux.withgradient(policymodel.model, states, actions, discounts) do modelchain, s, a, d
        pdist = modelchain(s) |> Flux.cpu 
        ent_lpdf = hcat([[Distributions.entropy(coldist), log(coldist[colact])] for (coldist, colact) in zip(eachcol(pdist), a)]...) #@inbounds Distributions.logpdf.(pdist, actions)

        entropyloss = - Statistics.mean(ent_lpdf[1, :])
        policyloss = - Statistics.mean(d .* value_error .* ent_lpdf[2,:]) 

        return policyloss + β * entropyloss 
    end

    if !isfinite(vval)
        @warn "Value loss is $vval"
    end

    Flux.update!(valuemodel.opt, valuemodel.model, vgrads[1])

    if !isfinite(pval)
        @warn "Policy loss is $pval"
    end

    Flux.update!(policymodel.opt, policymodel.model, pgrads[1])
end