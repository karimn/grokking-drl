struct εGreedyStrategy <: AbstractStrategy
    ε::Float16
end

function selectaction(strategy::εGreedyStrategy, m::AbstractModel, state; rng::AbstractRNG = GLOBAL_RNG, usegpu = true)
    qvalues = m(usegpu ? Flux.gpu(state) : state)
    qvalues = usegpu ? Flux.cpu(qvalues) : qvalues
    explored = false

    if rand(rng) > strategy.ε
        action = argmax(qvalues) 
    else
        action = rand(rng, Base.OneTo(length(qvalues)))
        explored = true
    end

    return action, explored
end

function evaluate_model(strategy::AbstractStrategy, m::AbstractModel, env::AbstractEnv; nepisodes = 1, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    rs = []

    for _ in 1:nepisodes
        reset!(env)
        s, d = state(env), false
        push!(rs, 0)

        while !d 
            a, _ = selectaction(strategy, m, s, rng = rng, usegpu = usegpu)
            env(a)
            s, r, d = state(env), reward(env), is_terminated(env)
            rs[end] += r
        end
    end

    return Statistics.mean(rs), Statistics.std(rs)
end

GreedyStrategy() = εGreedyStrategy(0.0)