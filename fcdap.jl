struct FCDAP <: AbstractPolicyModel
    model
    opt
    lossfn
end

function FCDAP(inputdim::Int, outputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), usegpu = true)
    return CreateSimpleFCModel(FCDAP, inputdim, outputdim, valueopt; hiddendims, actfn, lossfn, usegpu)
end

(m::FCDAP)(state) = m.model(state)

function fullpass(m::FCDAP, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    logits = m(usegpu ? Flux.gpu(state) : state)

    dist = Distributions.Categorical(softmax(logits))
    action = rand(rng, dist)
    logpa = @inbounds Distributions.logpdf(dist, action)  
    ent = Distribtions.entropy(dist)
    isexplore = action != argmax(logits)

    return action, isexplore, logpa, ent
end

function selectaction(m::FCDAP, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    @pipe m(usegpu ? Flux.gpu(state) : state) |> 
        Flux.cpu |> 
        Flux.softmax |> 
        Distributions.Categorical |> 
        rand(rng, _)
end

selectgreedyaction(m::FCDAP, state) = argmax(m(state))

function train!(m::M, states, actions, rewards, gamma) where M <: AbstractPolicyModel 
    T = length(states)
    discounts = gamma.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    val, grads = Flux.withgradient(m.model, states, actions, rewards) do modelchain, s, a, r
        lpdf = @pipe modelchain.(s) |> 
            Flux.cpu |> 
            Flux.softmax.(_) |> 
            Distributions.Categorical.(_) |> 
            @inbounds Distributions.logpdf.(_, actions)

        Statistics.mean(- discounts .* returns .* lpdf)
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