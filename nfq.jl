
mutable struct NFQ <: AbstractDRLAlgorithm
    hiddendims::Vector{Int}
    valueopt::Flux.Optimise.AbstractOptimiser
    trainstrategy::AbstractStrategy
    evalstrategy::AbstractStrategy
    batchsize::Int
    epochs::Int
    onlinemodel::Union{Nothing, FCQ}

    NFQ(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs) = new(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs, nothing) 
end

function initmodels!(agent::NFQ, sdims, nactions; usegpu = true) 
    agent.onlinemodel = FCQ(sdims, nactions, agent.valueopt, hiddendims = agent.hiddendims, usegpu = usegpu) 
end

optimizemodel!(agent::NFQ, experiences, gamma, step; usegpu = true) = optimizemodel!(agent.onlinemodel, experiences, agent.epochs, gamma, usegpu = usegpu)
