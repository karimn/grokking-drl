struct DoubleNetworkActorCriticModel{PM, VM} <: AbstractActorCriticModel where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel::PM
    valuemodel::VM
    policymodelopt
    valuemodelopt
end

function DoubleNetworkActorCriticModel{PM, VM}(ninputdims::Int, noutputdims::Int, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser;
                                               usegpu = true) where {PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel = PM(ninputdims, noutputdims, policyhiddendims; usegpu)
    valuemodel = VM(ninputdims, valuehiddendims; usegpu)

    DoubleNetworkActorCriticModel{PM, VM}(policymodel, valuemodel, Flux.setup(policyopt, policymodel), Flux.setup(valueopt, valuemodel))
end

𝒱(m::DoubleNetworkActorCriticModel, state) = 𝒱(m.valuemodel, state)
π(m::DoubleNetworkActorCriticModel, state) = π(m.policymodel, state)

selectaction(m::DoubleNetworkActorCriticModel, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) = selectaction(m.policymodel, state; rng, usegpu)
selectgreedyaction(m::DoubleNetworkActorCriticModel, state; usegpu = true) = selectgreedyaction(m.policymodel, state; usegpu)

function Flux.update!(::Nothing, m::DoubleNetworkActorCriticModel, grads::Tuple) 
    Flux.update!(m.policymodelopt, m.policymodel, grads[1])
    Flux.update!(m.valuemodelopt, m.valuemodel, grads[2])
end

function train!(m::DoubleNetworkActorCriticModel, states, actions, rewards, λ::Union{Nothing, Float32} = nothing, opt::Nothing = nothing; entropylossweight = 1.0, γ = 1.0, updatemodel = true)
    T = length(rewards) 
    discounts = γ.^range(0, T - 1)
    λ_discounts = (λ * γ).^range(0, T - 1) 
    returns = [sum(discounts[1:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    values = nothing 

    vval, vgrads = Flux.withgradient(m.valuemodel, states, returns) do valuemodel, s, r
        values = 𝒱(valuemodel, s) |> Flux.cpu |> vec 

        return ℒ(valuemodel, values, r[1:(end - 1)])
    end

    if λ ≡ nothing
        pop!(returns)

        Ψ = returns - values
    else
        push!(values, last(rewards))

        advs = rewards[1:(end - 1)] + γ * values[2:end] - values[1:(end - 1)]
        Ψ = [sum(λ_discounts[1:(T - t)] .* advs[t:end]) for t in 1:(T - 1)] 
    end

    pop!(discounts)

    pval, pgrads = Flux.withgradient(m.policymodel, states, actions, discounts, Ψ) do policymodel, s, a, d, Ψ 
        pdist = π(policymodel, s) |> Flux.cpu 
        ent_lpdf = reduce(hcat, [Distributions.entropy(coldist), log(coldist[colact])] for (coldist, colact) in zip(eachcol(pdist), a))

        entropyloss = - mean(ent_lpdf[1, :])
        policyloss = - mean(d .* Ψ .* ent_lpdf[2,:]) 

        return policyloss + entropylossweight * entropyloss 
    end

    isfinite(vval) || @warn "Value loss is $vval"
    isfinite(pval) || @warn "Policy loss is $pval"

    updatemodel && Flux.update!(m, (pgrads, vgrads))

    return Tuple(x for x in zip(pgrads, vgrads))
end

