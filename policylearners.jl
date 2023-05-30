function optimizemodel!(learner::L, states, actions, rewards; usegpu = true) where L <: AbstractPolicyLearner
    @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _) |> 
        train!(policymodel(learner), _, actions, rewards, opt(learner); γ = learner.γ)
end

function step!(learner::L, currstate; policymodel = policymodel(learner), env = environment(learner), rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action = selectaction(policymodel, currstate; rng, usegpu) |> only
    env(action)
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
        actions = Int[]
        rewards = Float32[]

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

function optimizemodel!(learner::Union{A3CLearner, VPGLearner}, states, actions, rewards, env::AbstractEnv = learner.env, localmodel::DoubleNetworkActorCriticModel = learner.model; λ = nothing, usegpu = true) 
    statesdata = @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _)

    laststate = state(env)
    failure = is_terminated(env) && !istruncated(env)

    nextvalue = 𝒱(localmodel, usegpu ? Flux.gpu(laststate) : laststate) |> Flux.cpu |> first
    push!(rewards, failure ? 0.0 : nextvalue)
        
    T = length(rewards) 
    discounts = learner.γ.^range(0, T - 1)
    returns = [sum(discounts[1:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    values = nothing 

    vval, vgrads = Flux.withgradient(learner.model, statesdata, returns) do valuemodel, s, r
        values = 𝒱(valuemodel, s) |> Flux.cpu |> vec 

        return ℒᵥ(valuemodel, values, r[1:(end - 1)])
    end

    isfinite(vval) || @warn "Value loss is $vval"

    if λ ≡ nothing
        pop!(returns)

        Ψ = returns - values
    else
        push!(values, last(rewards))

        λ_discounts = (λ * learner.γ).^range(0, T - 1) 

        advs = rewards[1:(end - 1)] + learner.γ * values[2:end] - values[1:(end - 1)]
        Ψ = [sum(λ_discounts[1:(T - t)] .* advs[t:end]) for t in 1:(T - 1)] 
    end

    pop!(discounts)

    pval, pgrads = Flux.withgradient(learner.model, statesdata, actions, discounts, Ψ) do policymodel, s, a, d, Ψ 
        pdist = π(policymodel, s) |> Flux.cpu 
        ent_lpdf = reduce(hcat, [Distributions.entropy(coldist), log(coldist[colact])] for (coldist, colact) in zip(eachcol(pdist), a))

        entropyloss = - mean(ent_lpdf[1, :])
        policyloss = - mean(d .* Ψ .* ent_lpdf[2,:]) 

        return policyloss + learner.β * entropyloss 
    end

    isfinite(pval) || @warn "Policy loss is $pval"

    try
        Flux.update!(learner.model, vgrads[1], pgrads[1])
    catch e
        throw(GradientException(learner.model, statesdata, actions, nothing, e, nothing, 1, length(rewards), (learner.λ * learner.γ).^range(0, length(rewards) - 1), grads))
    end
end