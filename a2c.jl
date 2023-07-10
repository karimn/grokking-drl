struct A2CLearner{E, M} <: AbstractPolicyLearner where {E <: AbstractAsyncEnv, M <: AbstractActorCriticModel}
    model::M
    modelopt
    epochs::Int
    env::E
    γ::Float32
    λ::Float32
    policylossweight::Float32
    valuelossweight::Float32
    entropylossweight::Float32
    max_nsteps::Int
    nworkers::Int
end

function A2CLearner{FCAC}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; 
                          nworkers, max_nsteps, policylossweight, valuelossweight, entropylossweight, λ, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv} 
    nS, nA = spacedim(env), nactions(env)
    policymodel = FCAC(nS, nA, hiddendims; nworkers, usegpu)

    return A2CLearner{E, FCAC}(policymodel, Flux.setup(opt, policymodel), epochs, env, γ, λ, policylossweight, valuelossweight, entropylossweight, max_nsteps, nworkers)
end

function A2CLearner{DoubleNetworkActorCriticModel{PM, VM}}(env::E, dnargs...; 
                                                   nworkers, max_nsteps, policylossweight, valuelossweight, entropylossweight, λ, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, PM, VM} 
    nS, nA = spacedim(env), nactions(env)
    model = DoubleNetworkActorCriticModel{PM, VM}(nS, nA, dnargs...; usegpu)

    return A2CLearner{E, DoubleNetworkActorCriticModel{PM, VM}}(model, nothing, epochs, env, γ, λ, policylossweight, valuelossweight, entropylossweight, max_nsteps, nworkers)
end


policymodel(l::A2CLearner) = l.model
environment(l::A2CLearner) = l.env
opt(l::A2CLearner) = l.modelopt

function _step!(learner::A2CLearner{E}, action; env) where E <: AbstractAsyncEnv
    env(action)
    newstate = Flux.cpu(state(env))

    return action, newstate, reward(env), is_terminated(env), istruncated(env)
end

function _step!(learner::A2CLearner{E}, action; env) where E <: AbstractEnv
    env(only(action))
    newstate = Flux.cpu.(state(env))

    return action, newstate, reward(env), is_terminated(env), istruncated(env)
end

function step!(learner::A2CLearner{E}, currstate::Vector; policymodel = policymodel(learner), env::E = environment(learner), rng = Random.GLOBAL_RNG, usegpu = true) where E <: AbstractAsyncEnv 
    _step!(learner, selectaction(policymodel, currstate; rng, usegpu); env)
end

function step!(learner::A2CLearner{E, FCAC}, currstate::Vector; policymodel = policymodel(learner), env::E = environment(learner), rng = Random.GLOBAL_RNG, usegpu = true) where E <: AbstractAsyncEnv 
    _step!(learner, selectaction(policymodel, reduce(hcat, currstate); rng, usegpu); env)
end

function train!(learner::A2CLearner; maxminutes::Int, maxepisodes::Int, goal_mean_reward, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    trainstart = now()

    running_timestep = zeros(Int, learner.nworkers)
    running_reward = zeros(Float32, learner.nworkers)
    running_seconds = repeat([trainstart], learner.nworkers)

    episodetimestep = Int[] 
    episodereward = Float32[]
    episodeseconds = [] 

    evalscores = Float32[] 
    results = EpisodeResult[]

    reset!(learner.env)
    currstate = state(learner.env)
    step, episode, n_steps_start = 0, 0, 0
    states, actions, rewards = [], [], []

    trainingdone = false

    while !trainingdone
        try
            step += 1

            push!(states, deepcopy(currstate))

            action, newstate, curr_reward, isterminal, _ = step!(learner, currstate; rng, usegpu) 

            push!(actions, copy(action))
            push!(rewards, copy(curr_reward))

            @assert length(actions) == (step - n_steps_start)
            @assert length(rewards) == (step - n_steps_start)

            running_timestep .+= 1
            running_reward .+= curr_reward 

            currstate = newstate

            if isterminal || step - n_steps_start >= learner.max_nsteps
                prevlearner = deepcopy(learner)
                prevcontext = (deepcopy(states), deepcopy(actions), deepcopy(rewards))

                try
                    optimizemodel!(learner, states, actions, rewards; usegpu) 
                catch e
                    throw(CorruptedNetworkException(prevlearner, prevcontext..., innerex = e))
                end

                any(p -> any(isnan, p), Flux.params(learner.model)) && throw(CorruptedNetworkException(prevlearner, prevcontext...))

                empty!(states) 
                empty!(actions)
                empty!(rewards)

                n_steps_start = step
            end

            if isterminal
                episodedone = now()
                evalscore, evalscoresd = evaluate(policymodel(learner), innerenv(learner.env); usegpu)

                terminatedidx = findall(is_terminateds(learner.env))

                reset!(learner.env, terminatedidx)
                newstate = state(learner.env)

                push!(episodetimestep, running_timestep[terminatedidx]...)
                push!(episodereward, running_reward[terminatedidx]...)
                push!(episodeseconds, (episodedone - running_seconds[terminatedidx])...)
                episode += length(terminatedidx) 
                push!(evalscores, evalscore)

                wallclockelapsed = now() - trainstart
                mean100evalscore = Statistics.mean(last(evalscores, 100))
                push!(results, EpisodeResult(sum(episodetimestep), mean(last(episodereward, 100)), mean100evalscore))

                @debug "Episode completed" terminatedidx episode evalscore evalscoresd mean100evalscore 

                reached_maxminutes = (wallclockelapsed.value / 60_000)  >= maxminutes * 60
                reached_maxepisodes = episode + learner.nworkers >= maxepisodes
                reached_goal_mean_reward = mean100evalscore >= goal_mean_reward 
                trainingdone = reached_maxminutes || reached_maxepisodes || reached_goal_mean_reward 

                running_timestep[terminatedidx] .= 0
                running_reward[terminatedidx] .= 0.0
                running_seconds[terminatedidx] .= now()
            end
        catch e
            rethrow()
        end
    end

    return results, evaluate(policymodel(learner), innerenv(learner.env); nepisodes = 100, usegpu)
end

function optimizemodel!(learner::A2CLearner{E}, states, actions, rewards; usegpu = true) where E <: AbstractAsyncEnv
    statesdata = @pipe mapreduce(t -> reduce(hcat, t), hcat, states) |> (usegpu ? Flux.gpu(_) : _) 

    laststates = reduce(hcat, state(learner.env))
    failures = is_terminateds(learner.env) .&& .!istruncateds(learner.env)

    nextvalues = 𝒱(learner.model, usegpu ? Flux.gpu(laststates) : laststates) |> Flux.cpu 
    push!(rewards, ifelse.(failures, zeros(Float32, length(failures)), nextvalues))
        
    return train!(learner.model, statesdata, actions, rewards, learner.λ, opt(learner); 
                  learner.γ, policylossweight = learner.policylossweight, valuelossweight = learner.valuelossweight, entropylossweight = learner.entropylossweight)
end

function optimizemodel!(learner::A2CLearner, states, actions, rewards; usegpu = true) 
    statesdata = reduce(hcat, states)

    laststates = state(learner.env)
    failure = is_terminated(learner.env) && !istruncated(learner.env)

    nextvalue = 𝒱(learner.model, usegpu ? Flux.gpu(laststates) : laststates) |> Flux.cpu 
    push!(rewards, failure ? Float32(0.0) : only(nextvalue))
        
    return train!(learner.model, statesdata, actions, rewards, learner.λ, opt(learner); 
                  learner.γ, policylossweight = learner.policylossweight, valuelossweight = learner.valuelossweight, entropylossweight = learner.entropylossweight)
end