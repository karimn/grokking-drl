function optimizemodel!(learner::L, states, actions, rewards; usegpu = true) where L <: AbstractPolicyLearner
    @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _) |> 
        train!(policymodel(learner), _, actions, rewards, opt(learner); γ = learner.γ)
end

function step!(learner::L, currstate; policymodel = policymodel(learner), env = environment(learner), rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action = selectaction(policymodel, currstate; rng, usegpu)
    env(only(action))
    newstate = Flux.cpu(state(env))

    return action, newstate, reward(env), is_terminated(env), istruncated(env)
end

function train!(learner::L; maxminutes::Int, maxepisodes::Int, goal_mean_reward, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractPolicyLearner}
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

        reset!(learner.env)

        currstate, isterminal = state(learner.env), is_terminated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while !isterminal 
            step += 1

            push!(states, copy(currstate))

            action, newstate, curr_reward, isterminal, _ = step!(learner, currstate; rng, usegpu) 

            push!(actions, action)
            push!(rewards, curr_reward)

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            currstate = newstate
        end

        optimizemodel!(learner, states, actions, rewards; usegpu)

        episode_elapsed = now() - episodestart
        trainingtime += episode_elapsed.value

        evalscore, evalscoresd = evaluate(policymodel(learner), learner.env; usegpu)
        push!(evalscores, evalscore)    
        
        mean100evalscores = mean(last(evalscores, 100))

        push!(results, EpisodeResult(sum(episodetimestep), mean(last(episodereward, 100)), mean100evalscores))

        @debug "Episode completed" episode = ep steps=step evalscore evalscoresd mean100evalscores 

        wallclockelapsed = now() - trainstart
        maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 
        rewardgoalreached = mean100evalscores >= goal_mean_reward

        if maxtimereached 
            @info "Maximum training time reached." 
            break
        end

        if rewardgoalreached
            @info "Reached goal mean reward" goal_mean_reward
            break
        end
    end

    return results, evaluate(policymodel(learner), env; nepisodes = 100, usegpu)
end

function optimizemodel!(learner::L, states, actions, rewards; usegpu = true) where L <: AbstractActorCriticLearner 
    optimizemodel!(model(learner), environment(learner), states, actions, rewards; γ = discount(learner), β = entropylossweight(learner), usegpu)
end
