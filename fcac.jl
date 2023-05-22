struct FCAC <: AbstractActorCriticModel
    model::Flux.Chain
    lossfn
    nworkers::Int
end

@functor FCAC (model,)

function FCAC(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; nworkers::Int = 1, actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdims => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Parallel(
            vcat,
            Flux.Chain(Flux.Dense(hiddendims[end] => outputdims), Flux.softmax),
            Flux.Dense(hiddendims[end] => 1)
        )
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCAC(modelchain, lossfn, nworkers)
end

(m::FCAC)(state) = m.model(state)
𝒱(m::FCAC, state) = m(state)[end, :]
π(m::FCAC, state) = m(state)[1:(end - 1), :]
ℒᵥ(m::FCAC, v̂, v) = m.lossfn(v̂, v)

function selectaction(m::FCAC, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu 

    try
        return rand.(rng, Distributions.Categorical.(copy(colp) for colp in eachcol(p)))
    catch e
        throw(NaNParamException(m, state))
    end
end

function selectgreedyaction(m::FCAC, state; usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return vec(getindex.(argmax(p, dims = 1), 1))
end

function ℒ(m::FCAC, states, actions, returns, rewards, discounts, λ_discounts; N, T, γ, valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0)
    output = m(states) |> Flux.cpu
    values = output[end, :]
    pdist = output[1:(end - 1), :]

    valueloss = ℒᵥ(m, values, returns[1:(end - N)])

    values = vcat(values, rewards[(end - N + 1):end])
    advs = reshape(rewards[1:(end - N)] + γ * values[(N + 1):end] - values[1:(end - N)], N, :)
    gaes = reduce(hcat, sum(λ_discounts[1:(T - t)]' .* advs[:, t:end], dims = 2) for t in 1:(T - 1)) 
    discounted_gaes = discounts' .* gaes 

    entropyloss = - mean(Distributions.entropy(coldist) for coldist in eachcol(pdist))

    lpdf = reshape([log(coldist[colact]) for (coldist, colact) in zip(eachcol(pdist), actions)], N, :)
    policyloss = - sum(lpdf .* discounted_gaes) / (N * (T - 1))

    return valuelossweight * valueloss + entropylossweight * entropyloss + policylossweight * policyloss 
end

function train!(m::FCAC, states, actions, rewards, λ, opt; γ, policylossweight = 1.0, valuelossweight = 1.0, entropylossweight = 1.0, updatemodel = true) 
    T = length(rewards)
    discounts = γ.^range(0, T - 1)
    λ_discounts = (λ * γ).^range(0, T - 1) 
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    pop!(discounts)

    N = m.nworkers  

    actions = reduce(vcat, actions)
    returns = reduce(vcat, returns)
    rewards = reduce(vcat, rewards)
    
    @assert size(states, 2) == length(actions) > 0 

    val, grads = Flux.withgradient(m -> ℒ(m, states, actions, returns, rewards, discounts, λ_discounts; N, T, γ, policylossweight, valuelossweight, entropylossweight), m)

    if !isfinite(val) 
        @warn "Value + policy + entropy loss is $val"
    end

    updatemodel && Flux.update!(opt, m, grads[1])

    return grads
end
