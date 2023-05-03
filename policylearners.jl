struct REINFORCELearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractPolicyBasedModel}
    policymodel::M
    epochs::Int
    env::E
end

function REINFORCELearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractPolicyBasedModel}
    nS, nA = spacedim(env), nactions(env)
    policymodel = M(nS, nA, opt; hiddendims, usegpu)

    return REINFORCELearner{E, M}(policymodel, epochs, env)
end

function optimizemodel!(learner::L, states, actions, rewards, gamma; usegpu = true) where L <: AbstractPolicyLearner
    train!(learner.policymodel, usegpu ? Flux.gpu(states) : states, actions, rewards, gamma)
end

function step!(learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action = selectaction(learner.policymodel, usegpu ? Flux.gpu(currstate) : currstate)
    learner.env(action)
    newstate = Flux.cpu(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env)
end

function train!(learner::L, gamma::Float64, maxminutes::Int, maxepisodes::Int; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractPolicyLearner}
    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    for ep in 1:maxepisodes
        states = []
        actions = []
        rewards = []

        reset!(learner.env)

        currstate, isterminal = Vector{Float32}(state(learner.env)), is_terminated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while true
            step += 1

            push!(states, currstate)

            action, newstate, curr_reward, isterminal = step!(learner, currstate; rng, usegpu) 

            push!(actions, action)
            push!(rewards, curr_reward)

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            currstate = newstate

            isterminal && break
        end

        optimizemodel!(learner, states, actions, rewards, gamma; usegpu)

        evalscore, _ = evaluate(learner.policymodel, env; usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))
    end

    return results, evaluate(learner.policymodel, env; nepisodes = 100, usegpu)
end