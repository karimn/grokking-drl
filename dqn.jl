mutable struct DQN <: AbstractDRLAlgorithm
    hiddendims::Vector{Int}
    valueopt::Flux.Optimise.AbstractOptimiser
    trainstrategy::AbstractStrategy
    evalstrategy::AbstractStrategy
    batchsize::Int
    epochs::Int
    updatemodelsteps::Int
    targetmodel::Union{Nothing, FCQ}
    onlinemodel::Union{Nothing, FCQ}

    DQN(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs, updatemodelsteps) = new(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs, updatemodelsteps, nothing, nothing) 
end

function optimizemodel!(agent::DQN, experiences, gamma, step; usegpu = true) 
    optimizemodel!(agent.onlinemodel, experiences, agent.epochs, gamma, targetmodel = agent.targetmodel, usegpu = usegpu)

    if (step % agent.updatemodelsteps) == 0 
        agent.targetmodel = deepcopy(agent.onlinemodel) 
    end
end

function initmodels!(agent::DQN, sdims, nactions; usegpu = true) 
    agent.onlinemodel = FCQ(sdims, nactions, agent.valueopt, hiddendims = agent.hiddendims, usegpu = usegpu) 
    agent.targetmodel = FCQ(sdims, nactions, agent.valueopt, hiddendims = agent.hiddendims, usegpu = usegpu) 
end
