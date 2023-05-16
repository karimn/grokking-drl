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

policymodel(l::REINFORCELearner) = l.policymodel

function step!(learner::L, currstate; policymodel = policymodel(learner), env = learner.env, rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractPolicyLearner
    action = selectaction(policymodel, currstate; rng, usegpu)
    env(action)
    newstate = Flux.cpu(state(env))

    return action, newstate, reward(env), is_terminated(env), istruncated(env)
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

        evalscore, evalscoresd = evaluate(policymodel(learner), env; usegpu)
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

    return results, evaluate(policymodel(learner), env; nepisodes = 100, usegpu)
end

struct VPGLearner{E, M} <: AbstractActorCriticLearner where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    model::M
    epochs::Int
    env::E
    γ::Float32
    β::Float32
end

function VPGLearner{M}(env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
                       β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, policyhiddendims, valuehiddendims, policyopt, valueopt; usegpu)

    return VPGLearner{E, M}(model, epochs, env, γ, β)
end

# function VPGLearner(::Type{M}, env::E, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser; 
#                        β, γ = Float32(1.0), epochs::Int = 1, usegpu = true) where {E <: AbstractEnv, M <: AbstractActorCriticModel}
#     return A3CLearner{M}(env, policyhiddendims, valuehiddendims, policyopt, valueopt,; max_nsteps = Inf, nworkers = 1, β, usegpu)
# end

environment(m::VPGLearner) = m.env
model(m::VPGLearner) = m.model
policymodel(m::VPGLearner) = model(m)
discount(m::VPGLearner) = m.γ 
entropylossweight(m::VPGLearner) = m.β

function optimizemodel!(model::M, env::AbstractEnv, states, actions, rewards; γ, β, λ = nothing, updatemodels = true, usegpu = true) where {M <: AbstractActorCriticModel}
    statesdata = @pipe hcat(states...) |> 
        (usegpu ? Flux.gpu(_) : _)

    laststate = state(env)
    failure = is_terminated(env) && !istruncated(env)

    nextvalue = value(model, usegpu ? Flux.gpu(laststate) : laststate) |> Flux.cpu |> first
    push!(rewards, failure ? 0.0 : nextvalue)
        
    return λ ≢ nothing ? train!(model, statesdata, actions, rewards, λ; γ, β, updatemodels) : train!(model, statesdata, actions, rewards; γ, β, updatemodels)
end

function optimizemodel!(learner::L, states, actions, rewards; usegpu = true) where L <: AbstractActorCriticLearner 
    optimizemodel!(model(learner), environment(learner), states, actions, rewards; γ = discount(learner), β = entropylossweight(learner), usegpu)
end
