update!(::B, ::Vector{Int}, ::Vector{Float32}) where {B <: AbstractBuffer} = nothing
Base.length(b::B) where B <: AbstractBuffer = b.currsize

mutable struct Buffer <: AbstractBuffer 
    maxsize::Int
    idx::Int
    currsize::Int
    buffer::DataFrame

    Buffer(maxsize::Int) = new(maxsize, 1, 0, DataFrame())
end

function store!(b::Buffer, s) 
    if b.currsize >= b.maxsize 
        b.buffer = DataFrame() # Let's clear out all the old experiences
    end

    push!(b.buffer, s)

    b.idx += 1
    b.currsize += 1 
end

readybatch(b::Buffer) = b.currsize == b.maxsize 

function getbatch(b::Buffer) 
    return collect(1:b.currsize), repeat([1/b.currsize], b.currsize), b.buffer
end

mutable struct ReplayBuffer <: AbstractBuffer 
    maxsize::Int
    batchsize::Int
    warmupbatches::Int
    idx::Int
    currsize::Int
    buffer::DataFrame

    ReplayBuffer(maxsize::Int, batchsize::Int, warmupbatches::Int = 5) = new(maxsize, batchsize, warmupbatches, 1, 0, DataFrame())
end

function store!(b::ReplayBuffer, s) 
    if b.currsize < b.maxsize 
        push!(b.buffer, s)
    else
        b.buffer[b.idx, :] = s 
    end

    b.idx = (b.idx % b.maxsize) + 1
    b.currsize = min(b.currsize + 1, b.maxsize)
end

function getbatch(b::ReplayBuffer; batchsize = b.batchsize)
    idxs = sample(1:nrow(b.buffer), batchsize, replace = false)

    return idxs, repeat([1 / batchsize], batchsize), b.buffer[idxs, :]
end

readybatch(b::ReplayBuffer) = b.currsize >= b.warmupbatches * b.batchsize 

mutable struct PrioritizedReplayBuffer <: AbstractBuffer
    maxsize::Int
    batchsize::Int
    warmupbatches::Int
    idx::Int
    currsize::Int
    buffer::DataFrame
    alpha::Float32
    beta::Float32
    betarate::Float32
    rankbased::Bool

    function PrioritizedReplayBuffer(maxsize::Int, batchsize::Int, warmupbatches::Int = 5; alpha::Float32, beta::Float32, betarate::Float32, rankbased = false) 
        new(maxsize, batchsize, warmupbatches, 1, 0, DataFrame(), alpha, beta, betarate, rankbased) 
    end
end

function store!(b::PrioritizedReplayBuffer, s)
    s = (s..., priority = b.currsize > 0 ? maximum(b.buffer.priority) : 1.0)

    if b.currsize < b.maxsize 
        push!(b.buffer, s)
    else
        b.buffer[b.idx, :] = s 
    end

    b.idx = (b.idx % b.maxsize) + 1
    b.currsize = min(b.currsize + 1, b.maxsize)
end

function update!(b::PrioritizedReplayBuffer, idxs::Vector{Int}, tderrors::Vector{Float32})
    @assert length(idxs) == length(tderrors)

    b.buffer.priority[idxs] = abs.(tderrors)
    
    b.rankbased && sort!(b.buffer, :priority, rev = true)
end

readybatch(b::PrioritizedReplayBuffer) = b.currsize >= b.warmupbatches * b.batchsize 

function getbatch(b::PrioritizedReplayBuffer; batchsize = b.batchsize)
    b.beta = min(1.0, b.beta / b.betarate)

    priorities = b.rankbased ? 1 ./ (collect(range(1, b.currsize)) .+ 1) : b.buffer.priority .+ 1e-6
    scaled_priorities = priorities.^b.alpha
    probs = scaled_priorities ./ sum(scaled_priorities)
    weights = (b.currsize .* probs).^(-b.beta)
    normalized_weights = weights ./ maximum(weights)
    idxs = wsample(range(1, b.currsize), normalized_weights, b.currsize, replace = false) 

    return idxs, normalized_weights, b.buffer[idxs, :]
end

mutable struct EpisodeBuffer <: AbstractBuffer
    env::AbstractAsyncEnv
    #buffer::DataFrame
    states::Vector 
    actions::Vector 
    returns::Vector{Vector{Float32}}
    logπ::Vector{Vector{Float32}}
    gaes::Vector{Vector{Float32}}
    maxepisodes::Int
    maxepisodesteps::Int
    γ::Float32
    λ::Float32
    episodesteps::Vector{Int}
    episoderewards::Vector{Float32}
    current_ep_idx::Vector{Int}

    function EpisodeBuffer(env::AbstractAsyncEnv, maxepisodes::Int, maxepisodesteps::Int; γ::Float32, λ::Float32) 
        newbuff = new(env, [], [], [], [], [], maxepisodes, maxepisodesteps, γ, λ, [], [], []) |> clear!
        clear!(newbuff)
    end
end

environment(b::EpisodeBuffer) = b.env
buffer_nworkers(b::EpisodeBuffer) = env_nworkers(b.env)
buffer_γ(b::EpisodeBuffer) = b.γ
buffer_maxepisodes(b::EpisodeBuffer) = b.maxepisodes

function clear!(b::EpisodeBuffer)
    n = buffer_nworkers(b)

    b.episodesteps = zeros(eltype(b.episodesteps), n)
    b.episoderewards = zeros(eltype(b.episoderewards), n)
    b.current_ep_idx = collect(1:n)
    b.states = [[] for _ in 1:n] 
    b.actions = [[] for _ in 1:n] 
    b.returns = [Float32[] for _ in 1:n] 
    b.logπ = [Float32[] for _ in 1:n] 
    b.gaes = [Float32[] for _ in 1:n]

    return b
end

function append_episode!(b::EpisodeBuffer, n::Int)
    append!(b.states, [[] for _ in 1:n])
    append!(b.actions, [[] for _ in 1:n])
    append!(b.returns, [Float32[] for _ in 1:n])
    append!(b.logπ, [Float32[] for _ in 1:n]) 
    append!(b.gaes, [Float32[] for _ in 1:n])
    append!(b.episodesteps, zeros(Int, n))
    append!(b.episoderewards, zeros(Float32, n))
end

function fill!(b::EpisodeBuffer, policymodel::AbstractPolicyModel, valuemodel::AbstractValueModel; rng::Random.AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    reset!(b.env)
    states = copy(state(b.env)) 

    nworkers = buffer_nworkers(b)

    worker_steps = zeros(Int, nworkers)
    worker_rewards = [Float32[] for _ in 1:nworkers] 

    bufferfull = false

    while !bufferfull && count(>(0), b.episodesteps) < b.maxepisodes / 2 
        actions, logπ = @pipe reduce(hcat, states) |> 
            (usegpu ? Flux.gpu(_) : _) |> 
            rand(rng, policymodel, _) |> 
            Flux.cpu  
        
        b.env(actions)
        nextstates, rewards, terminals, truncated = copy(state(b.env)), copy(reward(b.env)), is_terminateds(b.env), istruncateds(b.env)

        push!.(b.states[b.current_ep_idx], states)
        push!.(b.actions[b.current_ep_idx], actions)
        push!.(b.logπ[b.current_ep_idx], logπ)
        push!.(worker_rewards, rewards)

        worker_steps .+= 1
        @. terminals |= worker_steps >= b.maxepisodesteps 

        if any(terminals)
            terminatedidx = findall(terminals)
            nextvalues = zeros(nworkers)

            if any(truncated)
                truncatedidx = findall(truncated)
                nextvalues[truncatedidx] = @pipe reduce(hcat, nextstates[truncatedidx]) |> 
                    (usegpu ? Flux.gpu(_) : _) |> 
                    𝒱(valuemodel, _) |> 
                    vec |> 
                    Flux.cpu
            end

            reset!(b.env, terminatedidx)
            states = copy(state(b.env))

            eidx = b.current_ep_idx[terminatedidx]
            b.episodesteps[eidx] = worker_steps[terminatedidx] 
            b.episoderewards[eidx] = sum.(worker_rewards[terminatedidx])
            eprewards = vcat.(worker_rewards[terminatedidx], nextvalues[terminatedidx])
            b.returns[eidx] = accumulate_right.((v′, r) -> r + b.γ * v′, eprewards)

            #epstates = @pipe b.states[eidx] |> (usegpu ? Flux.gpu(_) : _)
            #epvalues = vec.(𝒱.(Ref(valuemodel), reduce.(hcat, epstates))) |> Flux.cpu 
            epvalues = [Flux.cpu(vec(𝒱(valuemodel, reduce(hcat, usegpu ? Flux.gpu(epstates) : epstates)))) for epstates in b.states[eidx]]
            append!.(epvalues, nextvalues[terminatedidx])
            deltas = [epr[1:(end - 1)] + b.γ * epv[2:end] - epv[1:(end - 1)] for (epr, epv) in zip(eprewards, epvalues)]
            b.gaes[eidx] = accumulate_right.((nextgae, delta) -> delta + b.γ * b.λ * nextgae, deltas)

            empty!.(worker_rewards[terminatedidx])
            worker_steps[terminatedidx] .= 0

            new_ep_idx = range(maximum(b.current_ep_idx) + 1, length = length(terminatedidx))

            if maximum(new_ep_idx) >= b.maxepisodes
                bufferfull = true
            else
                b.current_ep_idx[terminatedidx] = new_ep_idx
                append_episode!(b, length(terminatedidx))
            end
        else
            states = copy(nextstates)
        end
    end


    emptyepisodes = findall(b.episodesteps .== 0)

    deleteat!(b.states, emptyepisodes)
    deleteat!(b.actions, emptyepisodes)
    deleteat!(b.returns, emptyepisodes)
    deleteat!(b.logπ, emptyepisodes)
    deleteat!(b.gaes, emptyepisodes)

    deleteat!(b.episodesteps, emptyepisodes)
    deleteat!(b.episoderewards, emptyepisodes)

    #@debug "Buffer filled." episodes=length(b.episodesteps) empty=length(emptyepisodes) steps=sum(b.episodesteps)

    return b.episodesteps, b.episoderewards 
end

nsamples(b::EpisodeBuffer) = sum(length, b.actions)

getstacks(b::EpisodeBuffer) = mapreduce(episodestates -> reduce(hcat, episodestates), hcat, b.states), reduce(vcat, b.actions), reduce(vcat, b.returns), reduce(vcat, b.logπ), reduce(vcat, b.gaes)