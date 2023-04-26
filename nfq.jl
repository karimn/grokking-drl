
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

function train!(agent::A, env::AbstractEnv, gamma::Float64, maxminutes::Int, maxepisodes::Int; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where A <: AbstractDRLAlgorithm
    nS, nA = spacedim(env), nactions(env)

    initmodels!(agent, nS, nA, usegpu = usegpu)

    episodereward = Float64[]
    episodetimestep = Int[]
    experiences = []

    for _ in 1:maxepisodes
        reset!(env)

        currstate, isterminal = state(env), is_terminated(env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)

        step = 0

        while true
            step += 1

            # Interaction step
            action = selectaction(agent.trainstrategy, agent.onlinemodel, currstate, rng = rng, usegpu = usegpu)
            env(action)
            newstate = state(env)
            episodereward[end] += curr_reward = reward(env)
            episodetimestep[end] += 1
            isterminal = is_terminated(env)
            istruncated = false # is_truncated(env)

            push!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !istruncated))
            currstate = newstate

            if length(experiences) >= agent.batchsize
                optimizemodel!(agent, experiences, gamma, step, usegpu = usegpu)
                empty!(experiences) 
            end

            isterminal && break
        end
    end

    evaluate_model(agent.evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
end