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
    Flux.update!(m, grads)
end

function Flux.update!(m::DoubleNetworkActorCriticModel, grads::Tuple) 
    Flux.update!(m, grads[1], grads[2])
end

function Flux.update!(m::DoubleNetworkActorCriticModel, pgrads, vgrads) 
    Flux.update!(m.policymodelopt, m.policymodel, pgrads[1])
    Flux.update!(m.valuemodelopt, m.valuemodel, vgrads[1])
end

function train!(m::DoubleNetworkActorCriticModel, states, actions, rewards, λ::Union{Nothing, Float32} = nothing, opt::Nothing = nothing; valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0, γ = 1.0) 
    flatrewards = reduce(vcat, rewards)
    flatactions = reduce(vcat, actions)

    @assert length(flatrewards) == length(rewards) && length(flatactions) == length(actions) "Only handling a single environment"

    train!(m, states, flatactions, flatrewards, λ; valuelossweight, policylossweight, entropylossweight, γ)
end

function Flux.withgradient(m::DoubleNetworkActorCriticModel, states, actions, rewards::Vector{R}, λ::Union{Nothing, Float32} = nothing; valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0, γ = 1.0) where R <: Real
    T = length(rewards) 
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[1:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    values = nothing 

    vval, vgrads = Flux.withgradient(m.valuemodel, states, returns) do valuemodel, s, r
        values = 𝒱(valuemodel, s) |> Flux.cpu |> vec 

        return valuelossweight * ℒ(valuemodel, values, r[1:(end - 1)])
    end

    isfinite(vval) || @warn "Value loss is $vval"

    if λ ≡ nothing
        pop!(returns)

        Ψ = returns - values
    else
        push!(values, last(rewards))

        λ_discounts = (λ * γ).^range(0, T - 1) 

        advs = rewards[1:(end - 1)] + γ * values[2:end] - values[1:(end - 1)]
        Ψ = [sum(λ_discounts[1:(T - t)] .* advs[t:end]) for t in 1:(T - 1)] 
    end

    pop!(discounts)

    pval, pgrads = Flux.withgradient(m.policymodel, states, actions, discounts, Ψ) do policymodel, s, a, d, Ψ 
        pdist = π(policymodel, s) |> Flux.cpu 
        ent_lpdf = reduce(hcat, [Distributions.entropy(coldist), log(coldist[colact])] for (coldist, colact) in zip(eachcol(pdist), a))

        entropyloss = - mean(ent_lpdf[1, :])
        policyloss = - mean(d .* Ψ .* ent_lpdf[2,:]) 

        return policylossweight * policyloss + entropylossweight * entropyloss 
    end

    isfinite(pval) || @warn "Policy loss is $pval"

    return Tuple(((pg,), (vg,)) for (pg, vg) in zip(pgrads, vgrads))
end

function train!(m::DoubleNetworkActorCriticModel, states, actions, rewards::Vector{R}, λ::Union{Nothing, Float32} = nothing; valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0, γ = 1.0) where R <: Real
    grads = Flux.withgradient(m, states, actions, rewards, λ; valuelossweight, policylossweight, entropylossweight, γ)  

    try
        Flux.update!(m, grads[1][1], grads[1][2])
    catch e
        throw(GradientException(m, states, actions, nothing, e, nothing, 1, length(rewards), (λ * γ).^range(0, length(rewards) - 1), (pgrads, vgrads)))
    end

    return grads 
end

