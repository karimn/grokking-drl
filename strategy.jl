struct εGreedyStrategy <: AbstractStrategy
    ε::Float16
end

function selectaction!(strategy::S, m::AbstractValueModel, state; rng::AbstractRNG = GLOBAL_RNG, usegpu = true) where S <: AbstractStrategy
    retvals = selectaction(strategy, m, state; rng, usegpu)
    decay!(strategy)

    return retvals
end

function selectaction(strategy::S, m::AbstractValueModel, state; rng::AbstractRNG = GLOBAL_RNG, usegpu = true) where S <: AbstractStrategy
    qvalues = @pipe Vector{Float32}(state) |> m(usegpu ? Flux.gpu(_) : _)
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

function evaluate(strategy::S, m::AbstractValueModel, env::AbstractEnv; nepisodes = 1, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where S <: AbstractStrategy
    rs = []
    steps = []

    for _ in 1:nepisodes
        reset!(env)
        s, d = Vector{Float32}(state(env)), false
        push!(rs, 0)
        push!(steps, 0)

        while !d 
            steps[end] += 1
            a, _ = selectaction(strategy, m, s, rng = rng, usegpu = usegpu)
            env(a)
            s, r, d = state(env), reward(env), is_terminated(env)
            rs[end] += r
        end
    end

    return Statistics.mean(rs), Statistics.std(rs), Statistics.mean(steps)
end

decay!(s::εGreedyStrategy) = s.ε

GreedyStrategy() = εGreedyStrategy(0.0)

mutable struct εGreedyLinearStrategy <: AbstractStrategy
    ε::Float16
    min_ε::Float16
    decaysteps::Int
    init_ε::Float64
    step::Int
end

εGreedyLinearStrategy(ε::Float64, min_ε::Float64, decaysteps::Int) = εGreedyLinearStrategy(ε, min_ε, decaysteps, ε, 0)

function decay!(strategy::εGreedyLinearStrategy)
    strategy.ε = 1 - strategy.step / strategy.decaysteps
    strategy.ε = clamp((strategy.init_ε - strategy.min_ε) * strategy.ε + strategy.min_ε, strategy.min_ε, strategy.init_ε)
    strategy.step += 1

    return strategy.ε
end 

mutable struct εGreedyExpStrategy <: AbstractStrategy
    ε::Float16
    min_ε::Float16
    decaysteps::Int
    init_ε::Float64
    step::Int
    decayed_ε::Vector{Float16}
end

function εGreedyExpStrategy(;ε::Float64, min_ε::Float64, decaysteps::Int)  
    decayed_ε = (0.01 ./ exp.(range(-2, 0, decaysteps)) .- 0.01) * (ε - min_ε) .+ min_ε
    εGreedyExpStrategy(ε, min_ε, decaysteps, ε, 0, decayed_ε)
end

function decay!(strategy::εGreedyExpStrategy)
    strategy.ε = strategy.step >= strategy.decaysteps ? strategy.min_ε : strategy.decayed_ε[strategy.step + 1] 
    strategy.step += 1

    return strategy.ε
end 
