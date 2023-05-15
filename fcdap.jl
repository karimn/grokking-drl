struct FCDAP <: AbstractPolicyModel
    model
    opt
end

function FCDAP(inputdim::Int, outputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, usegpu = true)
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

    opt = Flux.setup(valueopt, modelchain)

    return FCDAP(modelchain, opt)
end

(m::FCDAP)(state) = m.model(state)

# function fullpass(m::FCDAP, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
#     p = m(usegpu ? Flux.gpu(state) : state) |> Flux.cpu
#     dist = Distributions.Categorical(p)
#     action = rand(rng, dist)
#     logpa = @inbounds Distributions.logpdf(dist, action)  
#     ent = Distributions.entropy(dist)
#     isexplore = action != argmax(p)

#     return action, isexplore, logpa, ent
# end

function selectaction(m::FCDAP, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    p = m(usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    try
        @pipe Distributions.Categorical(p) |> rand(rng, _) 
    catch e
        print(Flux.params(m.model))

        rethrow()
    end
end

selectgreedyaction(m::FCDAP, state) = argmax(m(state))

function train!(m::M, states, actions, rewards; γ = 1.0) where M <: AbstractPolicyModel 
    T = size(states, 2)
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    val, grads = Flux.withgradient(m.model, states, actions, returns, discounts) do modelchain, s, a, r, d
        lpdf = @pipe modelchain(s) |> 
            log.(_) |> 
            Flux.cpu |> 
            [collpdf[colact] for (collpdf, colact) in zip(eachcol(_), a)] 
            #Distributions.Categorical.(_) |> 
            #@inbounds Distributions.logpdf.(_, actions)

        - Statistics.mean(d .* r .* lpdf)
    end

    if !isfinite(val)
        @warn "loss is $val"
    end

    Flux.update!(m.opt, m.model, grads[1])
end

function evaluate(m::M, env::AbstractEnv; nepisodes = 1, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where M <: AbstractPolicyModel
    rs = []

    for _ in 1:nepisodes
        reset!(env)
        s, d = Vector{Float32}(state(env)), false
        push!(rs, 0)

        while !d 
            a = selectaction(m, s; rng, usegpu)
            env(a)
            s, r, d = state(env), reward(env), is_terminated(env)
            rs[end] += r
        end
    end

    return Statistics.mean(rs), Statistics.std(rs)
end