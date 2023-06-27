struct SACLearner{E <: AbstractEnv} <: AbstractActorCriticLearner
    policymodel::FCGP
    targetvaluemodel::FCTQV
    onlinevaluemodel::FCTQV
    policymodelopt
    valuemodelopt
    αopt
    env::E
    targetentropy::Float32
    experiences::ReplayBuffer
    update_target_steps::Int
    τ::Float32
    fixedα::Bool
    #logα::Vector{Float32}
    α::Vector{Float32}

    function SACLearner(env::E, experiences::ReplayBuffer, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser, αopt::Flux.Optimise.AbstractOptimiser; 
                        α::Union{Float32, Nothing} = nothing, update_target_steps::Int, τ = 1.0, usegpu = true) where E <: AbstractEnv
        nS, nA = spacedim(env), nactions(env)
        policymodel = FCGP(nS, nA, policyhiddendims; usegpu)
        targetvaluemodel = FCTQV(nS, nA, valuehiddendims; usegpu)
        onlinevaluemodel = FCTQV(nS, nA, valuehiddendims; usegpu)

        update_target_model!(targetvaluemodel, onlinevaluemodel, τ = 1.0)

        targetentropy = -DomainSets.dimension(action_space(env))

        fixedα = α !== nothing
        α = !fixedα ? α = [0f0] : [α]
        #logα = fixedα ? [log(α)] : [0f0] 

        return new{E}(policymodel, targetvaluemodel, onlinevaluemodel, Flux.setup(policyopt, policymodel), Flux.setup(valueopt, onlinevaluemodel), Flux.setup(αopt, α), 
                      env, targetentropy, experiences, update_target_steps, τ, fixedα, α) 
    end
end

environment(l::SACLearner) = l.env
buffer(l::SACLearner) = l.experiences
readybatch(l::SACLearner) = readybatch(l.experiences)

function step!(learner::SACLearner, s; rng::Random.AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    a = readybatch(learner) ? vec(rand(rng, learner.policymodel, s; usegpu)[1]) : rand(action_space(learner.env))
    learner.env(a)
    s′ = state(learner.env)

    return a, s′, reward(learner.env), is_terminated(learner.env), istruncated(learner.env) 
end

function train!(learner::SACLearner; maxminutes::Int, maxepisodes::Int, goal_mean_reward, evalepisodes = 1, γ::Float32 = 1.0f0, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
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
            action, newstate, curr_reward, isterminal, truncated = step!(learner, currstate; rng, usegpu) 

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            store!(learner.experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !truncated))
            currstate = newstate

            if readybatch(learner) 
                optimizemodel!(learner, γ, sum(episodetimestep), usegpu = usegpu)
            end
        end

        episode_elapsed = now() - episodestart
        trainingtime += episode_elapsed.value

        if (ep % evalepisodes) == 0
            evalscore, evalscoresd, evalsteps = evaluate(learner.policymodel, learner.env; rng, usegpu)
            push!(evalscores, evalscore)     

            mean100evalscores = mean(last(evalscores, 100))

            push!(results, EpisodeResult(sum(episodetimestep), mean(last(episodereward, 100)), mean100evalscores)) 

            @debug "Episode completed" episode = ep steps=step evalscore evalscoresd evalsteps mean100evalscores α=only(learner.α)

            rewardgoalreached = mean100evalscores >= goal_mean_reward

            if rewardgoalreached
                @info "Reached goal mean reward" goal_mean_reward
                break
            end
        end

        wallclockelapsed = now() - trainstart
        maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 

        if maxtimereached
            @info "Maximum training time reached." 
            break
        end

    end

    return results, evaluate(learner.policymodel, learner.env; nepisodes = 100, rng, usegpu)
end

function ℒₚ(m, s, pre_â, â, α, q_sa; ε = 1f-6)
    policydist = π(m, s) 
    logπ_s = @pipe Distributions.logpdf.(policydist, pre_â) .- log.(clamp.(1 .- â.^2, 0, 1) .+ ε) |> sum(_, dims = 1) |> vec

    return mean(α * logπ_s .- q_sa)
end

function optimizemodel!(learner::SACLearner, γ, totalsteps; rng::Random.AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    _..., batch = getbatch(learner.experiences)
    a = @pipe reduce(hcat, batch.a) |> (usegpu ? Flux.gpu(_) : _) 

    s = @pipe reduce(hcat, batch.s) |> (usegpu ? Flux.gpu(_) : _)
    s′ = @pipe reduce(hcat, batch.sp) |> (usegpu ? Flux.gpu(_) : _)

    # Training Q

    â′, logπ_s′ = rand(rng, learner.policymodel, s′; usegpu)
    current_q_s′a′ = min𝒬(learner.targetvaluemodel, s′, â′) .- only(learner.α) * vec(logπ_s′)

    target_q_sa = @pipe batch.r + γ * current_q_s′a′ .* (.!batch.failure) |> (usegpu ? Flux.gpu(_) : _) 

    vval, vgrads = Flux.withgradient(ℒ, learner.onlinevaluemodel, s, a, target_q_sa) 
    isfinite(vval) || @warn "Value loss is $vval" target_q_sa learner.α

    Flux.update!(learner.valuemodelopt, learner.onlinevaluemodel, vgrads[1])

    # Training the policy

    logπ_s = nothing

    pval, pgrads = Flux.withgradient(learner.policymodel, learner.onlinevaluemodel) do pm, vm
        â, logπ_s = rand(rng, pm, s; usegpu) 
        current_q_sa = min𝒬(vm, s, â)

        mean(only(learner.α) * logπ_s .- current_q_sa)
    end

    isfinite(pval) || @warn "Policy loss is $pval"

    Flux.update!(learner.policymodelopt, learner.policymodel, pgrads[1])

    if !learner.fixedα
        α_target = logπ_s .+ learner.targetentropy 
        αval, αgrads = Flux.withgradient(α -> -mean(only(α) * α_target), learner.α)

        isfinite(αval) || @warn "α loss is $αval"

        Flux.update!(learner.αopt, learner.α, αgrads[1])
    end

    if (totalsteps % learner.update_target_steps) == 0 
        update_target_value_model!(learner)
    end
end

update_target_value_model!(l::SACLearner) = update_target_model!(l.targetvaluemodel, l.onlinevaluemodel, τ = l.τ)