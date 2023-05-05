function step!(learner::AbstractValueLearner, s::AbstractStrategy, currstate; rng = Random.GLOBAL_RNG, usegpu = true)
    action, _ = selectaction!(s, learner, currstate; rng, usegpu)
    learner.env(action)
    newstate = Vector{Float32}(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), istruncated(learner.env) 

end

function train!(learner::L, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy, experiences::B; 
                maxminutes::Int, maxepisodes::Int, γ::Float64 = 1.0, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractValueLearner, B <: AbstractBuffer}
    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    trainstart = now()

    for ep in 1:maxepisodes
        episodestart, trainingtime = now(), 0

        reset!(learner.env)

        currstate, isterminal = Vector{Float32}(state(learner.env)), is_terminated(learner.env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)
        push!(episodeexploration, 0)

        step = 0

        while true
            step += 1

            # Interaction step
            action, newstate, curr_reward, isterminal, istruncated = step!(learner, trainstrategy, currstate; rng, usegpu) 

            episodereward[end] += curr_reward 
            episodetimestep[end] += 1
            episodeexploration[end] += 1

            store!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !istruncated))
            currstate = newstate

            if readybatch(experiences) 
                optimizemodel!(learner, experiences, γ, step, usegpu = usegpu)
            end

            isterminal && break
        end

        episode_elapsed = now() - episodestart
        trainingtime += episode_elapsed.value

        evalscore, evalscoresd, evalsteps = evaluate(evalstrategy, learner, env; usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))

        @debug "Episode completed" episode = ep steps=step evalscore evalscoresd evalsteps

        wallclockelapsed = now() - trainstart
        maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 

        if maxtimereached
            @info "Maximum training time reached." 
            break
        end
    end

    return results, evaluate(evalstrategy, learner, env; nepisodes = 100, usegpu)
end

struct FQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueModel}
    onlinemodel::M
    epochs::Int
    env::E
end

function FQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; epochs::Int, lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), usegpu = true) where {E <: AbstractEnv, M <: AbstractValueModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return FQNLearner{E, M}(model, epochs, env)
end

optimizemodel!(learner::FQNLearner, experiences::B, γ, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(learner.onlinemodel, experiences, learner.epochs, γ, usegpu = usegpu)

selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractValueLearner = selectaction(trainstrategy, learner.onlinemodel, currstate; rng, usegpu)
selectaction!(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractValueLearner = selectaction!(trainstrategy, learner.onlinemodel, currstate; rng, usegpu)

evaluate(evalstrategy::AbstractStrategy, learner::L, env::E; nepisodes = 1, usegpu = true) where {E <: AbstractEnv, L <: AbstractValueLearner} = evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)

(learner::FQNLearner)(state) = learner.onlinemodel(state)

save(learner::L, filename) where L <: AbstractValueLearner = save(learner.onlinemodel, filename)

mutable struct DQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueModel}
    onlinemodel::M
    targetmodel::M
    epochs::Int
    env::E
    updatemodelsteps::Int
    isdoublelearner::Bool
    τ::Float32
end

function DQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser; 
                       epochs::Int, updatemodelsteps::Int, lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), isdouble = false, τ = 1.0, usegpu = true) where {E <: AbstractEnv, M <: AbstractValueModel}

    nS, nA = spacedim(env), nactions(env)
    onlinemodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)
    targetmodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return DQNLearner{E, M}(onlinemodel, targetmodel, epochs, env, updatemodelsteps, isdouble, τ)
end

function updatemodels!(l::DQNLearner) 
    if l.τ == 1.0
        l.targetmodel = deepcopy(l.onlinemodel)
    else
        update!(l.targetmodel, l.onlinemodel, τ = l.τ)
    end
end

function optimizemodel!(learner::DQNLearner, experiences::B, γ, step; usegpu = true) where B <: AbstractBuffer 
    optimizemodel!(learner.onlinemodel, experiences, learner.epochs, γ, argmaxmodel = learner.isdoublelearner ? learner.onlinemodel : learner.targetmodel, targetmodel = learner.targetmodel, usegpu = usegpu)

    (step % learner.updatemodelsteps) == 0 && updatemodels!(learner)
end

(learner::DQNLearner)(state) = learner.onlinemodel(state)
