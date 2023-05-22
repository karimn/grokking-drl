struct FCDAP <: AbstractPolicyModel
    model
end

@functor FCDAP

function FCDAP(inputdim::Int, outputdim::Int, hiddendims::Vector{Int}; actfn = Flux.relu, usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdim => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Dense(hiddendims[end] => outputdim),
        Flux.softmax
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCDAP(modelchain)
end

π(m::FCDAP, state) = m.model(state)

function selectaction(m::FCDAP, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return rand.(rng, Distributions.Categorical.(copy(colp) for colp in eachcol(p)))
end

selectgreedyaction(m::FCDAP, state; usegpu = true) = argmax(π(m, usegpu ? Flux.gpu(state) : state))

function train!(m::FCDAP, states, actions, rewards, opt; γ = 1.0)
    T = size(states, 2)
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    val, grads = Flux.withgradient(m, states, actions, returns, discounts) do policymodel, s, a, r, d
        lpdf = @pipe π(policymodel, s) |> 
            log.(_) |> 
            Flux.cpu |> 
            [collpdf[colact] for (collpdf, colact) in zip(eachcol(_), a)] 
            #Distributions.Categorical.(_) |> 
            #@inbounds Distributions.logpdf.(_, actions)

        - mean(d .* r .* lpdf)
    end

    !isfinite(val) && @warn "loss is $val"

    Flux.update!(opt, m, grads[1])
end

function evaluate(m::M, env::AbstractEnv; nepisodes = 1, greedy = true, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where M <: Union{AbstractPolicyModel, AbstractActorCriticModel}
    rs = []

    env = deepcopy(env)

    for _ in 1:nepisodes
        reset!(env)
        s, d = state(env), false
        push!(rs, 0)

        while !d 
            a = greedy ? selectgreedyaction(m, s; usegpu) : selectaction(m, s; rng, usegpu)
            env(only(a))
            s, r, d = state(env), reward(env), is_terminated(env)
            rs[end] += r
        end
    end

    return mean(rs), std(rs)
end
