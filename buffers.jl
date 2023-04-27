mutable struct Buffer{MS} <: AbstractBuffer 
    idx::Int
    currsize::Int
    buffer::DataFrame
end

Buffer{MS}() where MS = Buffer{MS}(1, 0, DataFrame())

function store!(b::Buffer{MS}, s) where MS
    if b.currsize < MS 
        push!(b.buffer, s)
    else
        b.buffer = DataFrame(;s...) # Let's clear out all the old experiences
    end

    b.idx += 1
    b.currsize += 1 
end

Base.length(b::B) where B <: AbstractBuffer = b.currsize
readybatch(b::Buffer{MS}) where MS = b.currsize == MS 

function getbatch(b::Buffer) 
    return b.buffer
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

getbatch(b::ReplayBuffer{MS, BS}; batchsize = BS) where {MS, BS} = b.buffer[sample(1:nrow(b.buffer), batchsize, replace = false), :]

readybatch(b::ReplayBuffer{MS, BS}) where {MS, BS} = b.currsize >= 5 * BS  