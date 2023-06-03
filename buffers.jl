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