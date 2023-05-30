mutable struct Buffer{MS} <: AbstractBuffer 
    idx::Int
    currsize::Int
    buffer::DataFrame
end

Buffer{MS}() where MS = Buffer{MS}(1, 0, DataFrame())

update!(::B, ::Vector{Int}, ::Vector{Float32}) where {B <: AbstractBuffer} = nothing

function store!(b::Buffer{MS}, s) where MS
    if b.currsize >= MS 
        b.buffer = DataFrame() # Let's clear out all the old experiences
    end

    push!(b.buffer, s)

    b.idx += 1
    b.currsize += 1 
end

Base.length(b::B) where B <: AbstractBuffer = b.currsize
readybatch(b::Buffer{MS}) where MS = b.currsize == MS 

function getbatch(b::Buffer) 
    return collect(1:b.currsize), repeat([1/b.currsize], b.currsize), b.buffer
end

mutable struct ReplayBuffer{MS, BS} <: AbstractBuffer 
    idx::Int
    currsize::Int
    buffer::DataFrame
end

ReplayBuffer{MS, BS}() where {MS, BS} = ReplayBuffer{MS, BS}(1, 0, DataFrame())

function store!(b::ReplayBuffer{MS, BS}, s) where {MS, BS}
    if b.currsize < MS 
        push!(b.buffer, s)
    else
        b.buffer[b.idx, :] = s 
    end

    b.idx = (b.idx % MS) + 1
    b.currsize = min(b.currsize + 1, MS)
end

function getbatch(b::ReplayBuffer{MS, BS}; batchsize = BS) where {MS, BS} 
    idxs = sample(1:nrow(b.buffer), batchsize, replace = false)

    return idxs, repeat([1 / batchsize], batchsize), b.buffer[idxs, :]
end

readybatch(b::ReplayBuffer{MS, BS}) where {MS, BS} = b.currsize >= 5 * BS 

mutable struct PrioritizedReplayBuffer{MS, BS} <: AbstractBuffer
    idx::Int
    currsize::Int
    buffer::DataFrame
    alpha::Float32
    beta::Float32
    betarate::Float32
    rankbased::Bool
end

PrioritizedReplayBuffer{MS, BS}(;alpha::Float32, beta::Float32, betarate::Float32, rankbased = false) where {MS, BS} = PrioritizedReplayBuffer{MS, BS}(1, 0, DataFrame(), alpha, beta, betarate, rankbased) 

function store!(b::PrioritizedReplayBuffer{MS, BS}, s) where {MS, BS}
    s = (s..., priority = b.currsize > 0 ? maximum(b.buffer.priority) : 1.0)

    if b.currsize < MS 
        push!(b.buffer, s)
    else
        b.buffer[b.idx, :] = s 
    end

    b.idx = (b.idx % MS) + 1
    b.currsize = min(b.currsize + 1, MS)
end

function update!(b::PrioritizedReplayBuffer, idxs::Vector{Int}, tderrors::Vector{Float32})
    @assert length(idxs) == length(tderrors)

    b.buffer.priority[idxs] = abs.(tderrors)
    
    b.rankbased && sort!(b.buffer, :priority, rev = true)
end

readybatch(b::PrioritizedReplayBuffer{MS, BS}) where {MS, BS} = b.currsize >= 5 * BS 

function getbatch(b::PrioritizedReplayBuffer{MS, BS}; batchsize = BS) where {MS, BS} 
    b.beta = min(1.0, b.beta / b.betarate)

    priorities = b.rankbased ? 1 ./ (collect(range(1, b.currsize)) .+ 1) : b.buffer.priority .+ 1e-6
    scaled_priorities = priorities.^b.alpha
    probs = scaled_priorities ./ sum(scaled_priorities)
    weights = (b.currsize .* probs).^(-b.beta)
    normalized_weights = weights ./ maximum(weights)
    idxs = wsample(range(1, b.currsize), normalized_weights, b.currsize, replace = false) 

    return idxs, normalized_weights, b.buffer[idxs, :]
end