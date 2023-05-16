struct A3CLearner{E, M} <: AbstractPolicyLearner where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    model::M
    epochs::Int
    env::E
    γ::Float32
    β::Float32
    λ::Union{Nothing, Float32}
    max_nsteps::Int
    nworkers::Int
end

function A3CLearner{M}(env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                       max_nsteps, nworkers, β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, policyhiddendims, valuehiddendims, policyopt, valueopt; usegpu)

    return A3CLearner{E, M}(model, epochs, env, γ, β, nothing, max_nsteps, nworkers)
end

function GAELearner(::Type{M}, env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                     max_nsteps, nworkers, β, λ, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, policyhiddendims, valuehiddendims, policyopt, valueopt; usegpu)

    return A3CLearner{E, M}(model, epochs, env, γ, β, λ, max_nsteps, nworkers)
end

function train!(learner::A3CLearner; maxminutes::Int, maxepisodes::Int, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true)
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
            atomic_add!(nepisodes, 1)
            episodestart, trainingtime = now(), 0

            reset!(localenv)

            currstate, isterminal = state(localenv), is_terminated(localenv)

            push!(episodereward[workerid], 0)
            push!(episodetimestep[workerid], 0)
            push!(episodeexploration[workerid], 0)

            nstepstart, total_episode_steps, step = 0, 0, 0

            states, actions, rewards = [], [], []

            try
                while !isterminal 
                    step += 1

                    push!(states, copy(currstate))

                    action, newstate, curr_reward, isterminal, _ = step!(learner, currstate; policymodel = localmodel, env = localenv, rng, usegpu) 

                    push!(actions, action)
                    push!(rewards, curr_reward)

                    episodereward[workerid][end] += curr_reward 
                    episodetimestep[workerid][end] += 1
                    episodeexploration[workerid][end] += 1

                    currstate = newstate

                    if isterminal || (step - nstepstart >= learner.max_nsteps)
                        # isterminal && @debug "Terminal state reached." workerid episode = nepisodes[] steps=step 
                        # isterminal || @debug "Max steps reached." workerid episode = nepisodes[] steps=step 

                        localmodel = optimizemodel!(learner, localmodel, localenv, states, actions, rewards; usegpu)

                        states, actions, rewards = [], [], []

                        nstepstart = step
                    end
                end
            catch e
                throw(WorkerException(workerid, learner.model, localmodel, e))
            end

            episode_elapsed = now() - episodestart
            trainingtime += episode_elapsed.value

            evalscore, evalscoresd = evaluate(learner.model, localenv; usegpu)
            push!(evalscores[workerid], evalscore)     

            @debug "Episode completed" workerid episode = nepisodes[] steps=step evalscore evalscoresd 

            push!(results[workerid], EpisodeResult(sum(episodetimestep[workerid]), Statistics.mean(last(episodereward[workerid], 100)), Statistics.mean(last(evalscores[workerid], 100))))

            wallclockelapsed = now() - trainstart
            maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 

            if maxtimereached || nepisodes[] >= maxepisodes
                atomic_or!(getoutsignal, true)
            end
        end
    end

    return results, evaluate(learner.model, env; nepisodes = 100, usegpu)
end

function optimizemodel!(learner::A3CLearner, localmodel::M, env::AbstractEnv, states, actions, rewards; usegpu = true) where M <: AbstractActorCriticModel
    grads = optimizemodel!(localmodel, env, states, actions, rewards; γ = learner.γ, β = learner.β, λ = learner.λ, updatemodels = true, usegpu)

    # Asynchronous: Hog Wild!
    update!(learner.model, grads)

    return deepcopy(learner.model)   
end