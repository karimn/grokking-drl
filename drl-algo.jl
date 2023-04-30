struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

function train!(learner::L, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy, gamma::Float64, maxminutes::Int, maxepisodes::Int, ::Type{B}; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractLearner, B <: AbstractBuffer}
    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]
    experiences = B() 

    for ep in 1:maxepisodes

        reset!(learner.env)

        currstate, isterminal = Vector{Float32}(state(learner.env)), is_terminated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while true
            step += 1

            # Interaction step
            action, explored = selectaction(trainstrategy, learner, currstate, rng = rng, usegpu = usegpu)
            decay!(trainstrategy)
            learner.env(action)
            newstate = Vector{Float32}(state(learner.env))
            episodereward[end] += curr_reward = reward(learner.env)
            episodetimestep[end] += 1
            episodeexploration[end] += 1
            isterminal = is_terminated(learner.env)
            istruncated = false # is_truncated(env)

            store!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !istruncated))
            currstate = newstate

            if readybatch(experiences) 
                optimizemodel!(learner, experiences, gamma, step, usegpu = usegpu)
            end

            isterminal && break
        end

        evalscore, _ = evaluate(evalstrategy, learner, env, nepisodes = 100, usegpu = usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))
    end

    return results, evaluate(evalstrategy, learner, env, nepisodes = 100, usegpu = usegpu)
end