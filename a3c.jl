struct A3CLearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    model::M
    modelopt
    epochs::Int
    env::E
    γ::Float32
    β::Float32
    λ::Union{Nothing, Float32}
    max_nsteps::Int
    nworkers::Int
end

function A3CLearner{DoubleNetworkActorCriticModel{PM, VM}}(env::E, modelargs...; max_nsteps, nworkers, β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, PM, VM}
    nS, nA = spacedim(env), nactions(env)
    model = M{PM, VM}(nS, nA, modelargs...; usegpu)

    return A3CLearner{E, M}(model, nothing, epochs, env, γ, β, nothing, max_nsteps, nworkers)
end

function GAELearner(::Type{M}, env::E, hiddendims::Vector{Int}, modelopt::Flux.Optimise.AbstractOptimiser; 
                    max_nsteps, nworkers, β, λ, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, hiddendims; usegpu)

    return A3CLearner{E, M}(model, Flux.setup(modelopt, model), epochs, env, γ, β, λ, max_nsteps, nworkers)
end

function GAELearner(::Type{DoubleNetworkActorCriticModel{PM, VM}}, env::E, modelargs...; max_nsteps, nworkers, β, λ, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, PM, VM}
    nS, nA = spacedim(env), nactions(env)
    model = DoubleNetworkActorCriticModel{PM, VM}(nS, nA, modelargs...; usegpu)

    return A3CLearner{E, DoubleNetworkActorCriticModel{PM, VM}}(model, nothing, epochs, env, γ, β, λ, max_nsteps, nworkers)
end

function train!(learner::A3CLearner; maxminutes::Int, maxepisodes::Int, goal_mean_reward, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
    getoutsignal = Atomic{Bool}(false) 
    nepisodes = Atomic{Int}(0)

    evalscores = Vector{Vector{Float32}}(undef, learner.nworkers) 
    episodereward = Vector{Vector{Float32}}(undef, learner.nworkers)
    episodetimestep = Vector{Vector{Int}}(undef, learner.nworkers)
    episodeexploration = Vector{Vector{Int}}(undef, learner.nworkers)
    results = Vector{Vector{EpisodeResult}}(undef, learner.nworkers)

    trainstart = now()

    @threads for workerid in 1:learner.nworkers
        localenv = deepcopy(learner.env)
        localmodel = deepcopy(learner.model)

        evalscores[workerid] = Float32[]
        episodereward[workerid] = Float32[]
        episodetimestep[workerid] = Int[]
        episodeexploration[workerid] = Int[]
        results[workerid] = EpisodeResult[]

        step = 0

        while !getoutsignal[]
            episodestart, trainingtime = now(), 0

            reset!(localenv)

            currstate, isterminal = state(localenv), is_terminated(localenv)

            push!(episodereward[workerid], 0)
            push!(episodetimestep[workerid], 0)
            push!(episodeexploration[workerid], 0)

            nstepstart, total_episode_steps, step = 0, 0, 0

            states, actions, rewards = [], [], []

            #try
                while !isterminal 
                    step += 1

                    push!(states, copy(currstate))

                    action, newstate, curr_reward, isterminal, _ = step!(learner, currstate; policymodel = localmodel, env = localenv, rng, usegpu) 

                    push!(actions, only(action))
                    push!(rewards, curr_reward)

                    episodereward[workerid][end] += curr_reward 
                    episodetimestep[workerid][end] += 1
                    episodeexploration[workerid][end] += 1

                    currstate = newstate

                    if isterminal || (step - nstepstart >= learner.max_nsteps)
                        # isterminal && @debug "Terminal state reached." workerid episode = nepisodes[] steps=step 
                        # isterminal || @debug "Max steps reached." workerid episode = nepisodes[] steps=step 

                        if length(actions) > 1 
                            localmodel = optimizemodel!(learner, localmodel, localenv, states, actions, rewards; usegpu)
                        end

                        states, actions, rewards = [], [], []

                        nstepstart = step
                    end
                end
            #=catch e
                throw(WorkerException(workerid, learner.model, localmodel, e))
            end=#

            atomic_add!(nepisodes, 1)

            episode_elapsed = now() - episodestart
            trainingtime += episode_elapsed.value

            evalscore, evalscoresd = evaluate(learner.model, localenv; usegpu)
            push!(evalscores[workerid], evalscore)     

            mean100evalscore = mean(last(evalscores[workerid], 100))

            @debug "Episode completed" workerid episode = nepisodes[] steps=step evalscore evalscoresd  mean100evalscore

            push!(results[workerid], EpisodeResult(sum(episodetimestep[workerid]), mean(last(episodereward[workerid], 100)), mean100evalscore))

            wallclockelapsed = now() - trainstart
            maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 

            if maxtimereached || nepisodes[] >= maxepisodes || mean100evalscore >= goal_mean_reward
                atomic_or!(getoutsignal, true)
            end
        end
    end

    return results, evaluate(learner.model, env; nepisodes = 100, usegpu)
end

function optimizemodel!(learner::A3CLearner, localmodel::M, env::AbstractEnv, states, actions, rewards; usegpu = true) where M <: AbstractActorCriticModel
    statesdata = @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _)

    laststate = state(env)
    failure = is_terminated(env) && !istruncated(env)

    nextvalue = 𝒱(localmodel, usegpu ? Flux.gpu(laststate) : laststate) |> Flux.cpu |> first
    push!(rewards, failure ? 0.0 : nextvalue)
        
    grads = train!(localmodel, statesdata, actions, rewards, learner.λ, learner.modelopt; γ = learner.γ, entropylossweight = learner.β, updatemodel = true)

    # Asynchronous: Hog Wild!
    Flux.update!(learner.modelopt, learner.model, grads[1])

    return deepcopy(learner.model)   
end