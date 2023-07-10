struct FCAC <: AbstractActorCriticModel # Fully connected actor-critic
    model::Flux.Chain
    lossfn
    nworkers::Int
    outputdims::Int
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
            Flux.Chain(Flux.Dense(hiddendims[end] => outputdims), Flux.Parallel(vcat, Flux.softmax, Flux.logsoftmax)),
            Flux.Dense(hiddendims[end] => 1)
        )
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCAC(modelchain, lossfn, nworkers, outputdims)
end

(m::FCAC)(state) = m.model(state)
𝒱(m::FCAC, state) = m(state)[end, :]
π(m::FCAC, state) = m(state)[1:(m.outputdims), :]
logπ(m::FCAC, state) = m(state)[(m.outputdims + 1):(end - 1), :]
ℒᵥ(m::FCAC, v̂, v) = m.lossfn(v̂, v)

function ℒ(m::FCAC, states, actions, returns, rewards, discounts, λ_discounts; N, T, γ, valuelossweight = 1.0, policylossweight = 1.0, entropylossweight = 1.0)
    output = m(states) |> Flux.cpu
    prob = output[1:(m.outputdims), :]
    logprob = output[(m.outputdims + 1):(end - 1), :]
    values = output[end, :]

    valueloss = ℒᵥ(m, values, returns[1:(end - N)])

    entropyloss = - mean(- sum(prob .* logprob; dims = 1)) 

    discounted_gaes = Flux.ignore_derivatives() do 
        values = vcat(values, rewards[(end - N + 1):end])
        _, gaes = calcgaes(values, rewards, λ_discounts; N, γ)

        return discounts' .* gaes 
    end

    lpdf = reshape([lp[a] for (a, lp) in zip(actions, eachcol(logprob))], N, :)
    policyloss = - sum(lpdf .* discounted_gaes) / (N * (T - 1))

    return valuelossweight * valueloss + entropylossweight * entropyloss + policylossweight * policyloss 
end

function Flux.withgradient(m::FCAC, states, actions, rewards, λ, opt; γ = 1.0, policylossweight = 1.0, valuelossweight = 1.0, entropylossweight = 1.0) 
    T = length(rewards)
    discounts = γ.^range(0, T - 1)
    λ_discounts = (λ * γ).^range(0, T - 1) 
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    pop!(discounts)

    N = m.nworkers  

    flatactions = reduce(vcat, actions)
    flatreturns = reduce(vcat, returns)
    flatrewards = reduce(vcat, rewards)
    
    @assert size(states, 2) == length(flatactions) > 0 

    val, grads = NaN, nothing

    try
        val, grads = Flux.withgradient(m -> ℒ(m, states, flatactions, flatreturns, flatrewards, discounts, λ_discounts; N, T, γ, policylossweight, valuelossweight, entropylossweight), m)
    catch e
        throw(GradientException(m, states, actions, returns, e, nothing, N, T, λ_discounts, nothing))
    end

    if !isfinite(val) 
        @warn "Value + policy + entropy loss is $val"
    end

    return val, grads
end

function train!(m::FCAC, states, actions, rewards, λ, opt; γ = 1.0, policylossweight = 1.0, valuelossweight = 1.0, entropylossweight = 1.0, updatemodel = true) 
    _, grads = Flux.withgradient(m, states, actions, rewards, λ, opt; γ, policylossweight, valuelossweight, entropylossweight)

    try
        updatemodel && Flux.update!(opt, m, grads[1])
    catch e
        throw(GradientException(m, states, actions, nothing, e, nothing, m.nworkers, length(rewards), (λ * γ).^range(0, length(rewards) - 1), grads))
    end

    return grads
end
