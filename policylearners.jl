struct REINFORCELearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractPolicyModel}
    policymodel::M
    epochs::Int
    env::E
end

function REINFORCELearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractPolicyModel}
    nS, nA = spacedim(env), nactions(env)
    policymodel = M(nS, nA, opt; hiddendims, usegpu)

    return REINFORCELearner{E, M}(policymodel, epochs, env)
end

function optimizemodel!(learner::L, states, actions, rewards, logpas; γ = 1.0, usegpu = true) where L <: AbstractPolicyLearner
    @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _) |> 
        train!(learner.policymodel, _, actions, rewards, logpas; γ)
end

function step!(learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action, _, logpa, _ = fullpass(learner.policymodel, currstate; rng, usegpu)
    #action = selectaction(learner.policymodel, currstate; rng, usegpu)
    learner.env(action)
    newstate = Flux.cpu(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), logpa
end

function train!(learner::L; maxminutes::Int, maxepisodes::Int, γ::Float32 = Float32(1.0), rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractPolicyLearner}
    evalscores = []
    episodereward = Float32[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    trainstart = now()

    for ep in 1:maxepisodes
        episodestart, trainingtime = now(), 0

        states = []
        actions = []
        rewards = []
        logpas = []

        reset!(learner.env)

        currstate, isterminal = state(learner.env), is_terminated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while !isterminal 
            step += 1

            push!(states, copy(currstate))

            action, newstate, curr_reward, isterminal, logpa = step!(learner, currstate; rng, usegpu) 

            push!(actions, action)
            push!(rewards, curr_reward)
            push!(logpas, logpa)

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            currstate = newstate
        end

        optimizemodel!(learner, states, actions, rewards, logpas; γ, usegpu)

        episode_elapsed = now() - episodestart
        trainingtime += episode_elapsed.value

        evalscore, evalscoresd = evaluate(learner.policymodel, env; usegpu)
        push!(evalscores, evalscore)     
        

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))

        @debug "Episode completed" episode = ep steps=step evalscore evalscoresd 

        wallclockelapsed = now() - trainstart
        maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 

        if maxtimereached
            @info "Maximum training time reached." 
            break
        end
    end

    return results, evaluate(learner.policymodel, env; nepisodes = 100, usegpu)
end