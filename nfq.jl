
mutable struct NFQ <: AbstractDRLAlgorithm
    hiddendims::Vector{Int}
    valueopt::Flux.Optimise.AbstractOptimiser
    # trainstrategy::AbstractStrategy
    # evalstrategy::AbstractStrategy
    #batchsize::Int
    epochs::Int
    onlinemodel::Union{Nothing, FCQ}

    NFQ(hiddendims, valueopt, epochs) = new(hiddendims, valueopt, epochs, nothing) 
end

function initmodels!(agent::NFQ, sdims, nactions; usegpu = true) 
    agent.onlinemodel = FCQ(sdims, nactions, agent.valueopt, hiddendims = agent.hiddendims, usegpu = usegpu) 
end

optimizemodel!(agent::NFQ, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(agent.onlinemodel, experiences, agent.epochs, gamma, usegpu = usegpu)
