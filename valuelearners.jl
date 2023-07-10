function step!(learner::Union{AbstractValueLearner, AbstractActorCriticLearner}, s::AbstractStrategy, currstate; rng = Random.GLOBAL_RNG, usegpu = true)
    action, _ = selectaction(s, learner, currstate; rng, usegpu)
    decay!(s)
    learner.env(action)
    newstate = Vector{Float32}(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), istruncated(learner.env) 
end

function train!(learner::AbstractActorCriticLearner, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy; 
                maxminutes::Int, maxepisodes::Int, goal_mean_reward, γ::Float32 = 1.0f0, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    train!(learner, trainstrategy, evalstrategy, buffer(learner); maxminutes, maxepisodes, goal_mean_reward, γ, rng, usegpu) 
end

function train!(learner::Union{AbstractValueLearner, AbstractActorCriticLearner}, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy, experiences::AbstractBuffer; 
                maxminutes::Int, maxepisodes::Int, goal_mean_reward, γ::Float32 = 1.0f0, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    evalscores = []
    episodereward = Float32[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    trainstart = now()

    for ep in 1:maxepisodes
        episodestart, trainingtime = now(), 0

        reset!(learner.env)

        currstate, isterminal, truncated = Vector{Float32}(state(learner.env)), is_terminated(learner.env), istruncated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while !isterminal && !truncated 
            step += 1

            # Interaction step
            action, newstate, curr_reward, isterminal, truncated = step!(learner, trainstrategy, currstate; rng, usegpu) 

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            store!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !truncated))
            currstate = newstate

            if readybatch(experiences) 
                optimizemodel!(learner, experiences, γ, sum(episodetimestep), usegpu = usegpu)
            end
        end

        episode_elapsed = now() - episodestart
        trainingtime += episode_elapsed.value

        evalscore, evalscoresd, evalsteps = evaluate(evalstrategy, learner; usegpu)
        push!(evalscores, evalscore)     

        mean100evalscores = mean(last(evalscores, 100))

        push!(results, EpisodeResult(sum(episodetimestep), mean(last(episodereward, 100)), mean100evalscores)) 

        @debug "Episode completed" episode = ep steps=step evalscore evalscoresd evalsteps mean100evalscores

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

    return results, evaluate(evalstrategy, learner; nepisodes = 100, usegpu)
end

function selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractValueLearner 
    selectaction(trainstrategy, learner.onlinemodel, currstate; rng, usegpu)
end

function selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractActorCriticLearner 
    selectaction(trainstrategy, learner.onlinemodel, currstate; maxexploration = !readybatch(learner), rng, usegpu)
end

function evaluate(evalstrategy::AbstractStrategy, learner::L, env::E = environment(learner); nepisodes = 1, usegpu = true) where {E <: AbstractEnv, L <: Union{AbstractValueLearner, AbstractActorCriticLearner}} 
    evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)
end

function update_target_model!(l::L) where L <: Union{AbstractValueLearner, AbstractActorCriticLearner}
    if l.τ < 1.0
        update_target_model!(l.targetmodel, l.onlinemodel, τ = l.τ)
    else
        l.targetmodel = deepcopy(l.onlinemodel)
    end
end

save(learner::L, filename) where L <: Union{AbstractValueLearner, AbstractActorCriticLearner} = save(learner.onlinemodel, filename)