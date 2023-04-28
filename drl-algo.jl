struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

function train!(agent::A, env::AbstractEnv, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy, gamma::Float64, maxminutes::Int, maxepisodes::Int, ::Type{B}; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {A <: AbstractDRLAlgorithm, B <: AbstractBuffer}
    nS, nA = spacedim(env), nactions(env)

    initmodels!(agent, nS, nA, usegpu = usegpu)

    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]
    experiences = B() 

    for ep in 1:maxepisodes

        reset!(env)

        currstate, isterminal = Vector{Float32}(state(env)), is_terminated(env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while true
            step += 1

            # Interaction step
            action, explored = selectaction(trainstrategy, agent.onlinemodel, currstate, rng = rng, usegpu = usegpu)
            decay!(trainstrategy)
            env(action)
            newstate = Vector{Float32}(state(env))
            episodereward[end] += curr_reward = reward(env)
            episodetimestep[end] += 1
            episodeexploration[end] += 1
            isterminal = is_terminated(env)
            istruncated = false # is_truncated(env)

            store!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !istruncated))
            currstate = newstate

            if readybatch(experiences) 
                optimizemodel!(agent, experiences, gamma, step, usegpu = usegpu)
            end

            isterminal && break
        end

        evalscore, _ = evaluate(evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))
    end

    return results, evaluate(evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
end