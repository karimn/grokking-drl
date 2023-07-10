decay!(::AbstractStrategy) = nothing

struct εGreedyStrategy <: AbstractDiscreteStrategy
    ε::Float16
end

function selectaction(strategy::S, m::AbstractValueModel, state; rng::AbstractRNG = GLOBAL_RNG, usegpu = true) where S <: AbstractDiscreteStrategy
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

function evaluate(strategy::AbstractStrategy, m::AbstractModel, env::AbstractEnv; nepisodes = 1, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    rs = []
    steps = []

    env = deepcopy(env)

    for _ in 1:nepisodes
        reset!(env)
        s, d = Vector{Float32}(state(env)), false
        push!(rs, 0)
        push!(steps, 0)

        while !d 
            steps[end] += 1
            a, _ = selectaction(strategy, m, s, rng = rng, usegpu = usegpu)
            env(a)
            s, r, d = state(env), reward(env), is_terminated(env) || istruncated(env)
            rs[end] += r
        end
    end

    return mean(rs), std(rs), mean(steps)
end

decay!(s::εGreedyStrategy) = s.ε

GreedyStrategy() = εGreedyStrategy(0.0)

mutable struct εGreedyLinearStrategy <: AbstractDiscreteStrategy
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

mutable struct εGreedyExpStrategy <: AbstractDiscreteStrategy
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

struct ContinuousGreedyStrategy <: AbstractContinuousStrategy 
    low::Float32
    high::Float32
end

ContinuousGreedyStrategy() = ContinuousGreedyStrategy(-1.0f0, 1.0f0) 

function selectaction(s::ContinuousGreedyStrategy, m::AbstractModel, state; rng::Random.AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    greedyact = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu #|> only

    return clamp.(greedyact, s.low, s.high), false
end

struct NormalNoiseGreedyStrategy <: AbstractContinuousStrategy
    low::Float32
    high::Float32
    noise_ratio::Float32

    NormalNoiseGreedyStrategy(low, high, exploration_noise_ratio = 0.1) = new(low, high, exploration_noise_ratio)
    NormalNoiseGreedyStrategy(exploration_noise_ratio = 0.1) = new(-1.0, 1.0, exploration_noise_ratio)
end

mutable struct NormalNoiseDecayStrategy <: AbstractContinuousStrategy 
    t::Int
    low::Float32
    high::Float32
    noise_ratio::Float32
    init_noise_ratio::Float32
    min_noise_ratio::Float32
    decaysteps::Int

    function NormalNoiseDecayStrategy(low::Float32 = -1.0f0, high::Float32 = 1.0f0; init_noise_ratio = 0.5f0, min_noise_ratio = 0.1f0, decaysteps = 10_000) 
        @assert length(low) == length(high)

        new(0, low, high, init_noise_ratio, init_noise_ratio, min_noise_ratio, decaysteps)
    end
end 

function decay!(s::NormalNoiseDecayStrategy)
    s.noise_ratio = @pipe 1 - s.t / s.decaysteps |> 
        (s.init_noise_ratio - s.min_noise_ratio) * _ + s.min_noise_ratio |> 
        clamp(_, s.min_noise_ratio, s.init_noise_ratio)

    s.t += 1

    return s.noise_ratio
end

function selectaction(s::Union{NormalNoiseGreedyStrategy, NormalNoiseDecayStrategy}, m::AbstractPolicyModel, state; maxexploration = false, rng::Random.AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    noisescale = maxexploration ? s.high : s.noise_ratio * s.high

    greedyaction = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    noise = rand(rng, Distributions.Normal(0, noisescale), length(greedyaction))
    noisyaction = greedyaction .+ noise
    action = clamp.(noisyaction, s.low, s.high)
    ratio_noise_injected = mean(abs.((greedyaction .- action) / (s.high - s.low)))

    return action, ratio_noise_injected 
end
