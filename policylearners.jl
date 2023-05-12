struct REINFORCELearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractPolicyModel}
    policymodel::M
    epochs::Int
    env::E
    γ::Float32 
end

function REINFORCELearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractPolicyModel}
    nS, nA = spacedim(env), nactions(env)
    policymodel = M(nS, nA, opt; hiddendims, usegpu)

    return REINFORCELearner{E, M}(policymodel, epochs, env, γ)
end

function optimizemodel!(learner::L, states, actions, rewards; usegpu = true) where L <: AbstractPolicyLearner
    @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _) |> 
        train!(learner.policymodel, _, actions, rewards; γ = learner.γ)
end

function step!(learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action = selectaction(learner.policymodel, currstate; rng, usegpu)
    learner.env(action)
    newstate = Flux.cpu(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), istruncated(learner.env)
end

function train!(learner::L; maxminutes::Int, maxepisodes::Int, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractPolicyLearner}
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

struct VPGLearner{E, PM, VM} <: AbstractPolicyLearner where {E <: AbstractEnv, PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    policymodel::PM
    valuemodel::VM
    epochs::Int
    env::E
    γ::Float32
    β::Float32
end

function VPGLearner{PM, VM}(env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                            β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, PM <: AbstractPolicyModel, VM <: AbstractValueModel}
    nS, nA = spacedim(env), nactions(env)
    policymodel = PM(nS, nA, policyopt; hiddendims = policyhiddendims, usegpu)
    valuemodel = VM(nS, valueopt; hiddendims = valuehiddendims, usegpu)

    return VPGLearner{E, PM, VM}(policymodel, valuemodel, epochs, env, γ, β)
end

function optimizemodel!(learner::VPGLearner, states, actions, rewards; usegpu = true)
    statesdata = @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _)

    laststate, truncated = state(learner.env), istruncated(learner.env)

    nextvalue = learner.valuemodel(usegpu ? Flux.gpu(laststate) : laststate) |> Flux.cpu |> first
    push!(rewards, truncated ? nextvalue : 0.0)
        
    train!(learner.policymodel, learner.valuemodel, statesdata, actions, rewards; γ = learner.γ, β = learner.β)
end