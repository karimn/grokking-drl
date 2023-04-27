struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

function train!(agent::A, env::AbstractEnv, gamma::Float64, maxminutes::Int, maxepisodes::Int; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where A <: AbstractDRLAlgorithm
    nS, nA = spacedim(env), nactions(env)

    initmodels!(agent, nS, nA, usegpu = usegpu)

    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    for ep in 1:maxepisodes
        experiences = []

        reset!(env)

        currstate, isterminal = state(env), is_terminated(env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while true
            step += 1

            # Interaction step
            action, explored = selectaction(agent.trainstrategy, agent.onlinemodel, currstate, rng = rng, usegpu = usegpu)
            env(action)
            newstate = state(env)
            episodereward[end] += curr_reward = reward(env)
            episodetimestep[end] += 1
            episodeexploration[end] += 1
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

        evalscore, _ = evaluate_model(agent.evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))

    end

    return results, evaluate_model(agent.evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
end