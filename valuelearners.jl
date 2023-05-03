function step!(learner::AbstractValueLearner, s::AbstractStrategy, state; rng = Random.GLOBAL_RNG, usegpu = true)
    action, _ = selectaction!(s, learner, state; rng, usegpu)
    learner.env(action)
    newstate = Vector{Float32}(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), false

end

function train!(learner::L, trainstrategy::AbstractStrategy, evalstrategy::AbstractStrategy, gamma::Float64, maxminutes::Int, maxepisodes::Int, experiences::B; 
                rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where {L <: AbstractValueLearner, B <: AbstractBuffer}
    evalscores = []
    episodereward = Float64[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    for ep in 1:maxepisodes

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
                optimizemodel!(learner, experiences, gamma, step, usegpu = usegpu)
            end

            isterminal && break
        end

        evalscore, _ = evaluate(evalstrategy, learner, env, nepisodes = 100, usegpu = usegpu)
        push!(evalscores, evalscore)     

        push!(results, EpisodeResult(sum(episodetimestep), Statistics.mean(last(episodereward, 100)), Statistics.mean(last(evalscores, 100))))
    end

    return results, evaluate(evalstrategy, learner, env, nepisodes = 100, usegpu = usegpu)
end

struct FQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueBasedModel}
    onlinemodel::M
    epochs::Int
    env::E
end

function FQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int; lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), usegpu = true) where {E <: AbstractEnv, M <: AbstractValueBasedModel}
    nS, nA = spacedim(env), nactions(env)
    model = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return FQNLearner{E, M}(model, epochs, env)
end

optimizemodel!(learner::FQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer = optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, usegpu = usegpu)

selectaction(trainstrategy::AbstractStrategy, learner::L, currstate; rng = Random.GLOBAL_RNG, usegpu = true) where L <: AbstractValueLearner = selectaction(trainstrategy, learner.onlinemodel, currstate, rng = rng, usegpu = usegpu)

evaluate(evalstrategy::AbstractStrategy, learner::L, env::E; nepisodes = 1, usegpu = true) where {E <: AbstractEnv, L <: AbstractValueLearner} = evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)

(learner::FQNLearner)(state) = learner.onlinemodel(state)

save(learner::L, filename) where L <: AbstractValueLearner = save(learner.onlinemodel, filename)

mutable struct DQNLearner{E, M} <: AbstractValueLearner where {E <: AbstractEnv, M <: AbstractValueBasedModel}
    onlinemodel::M
    targetmodel::M
    epochs::Int
    env::E
    updatemodelsteps::Int
    isdoublelearner::Bool
    tau::Float32
end

function DQNLearner{M}(env::E, hiddendims::Vector{Int}, opt::Flux.Optimise.AbstractOptimiser, epochs::Int, updatemodelsteps::Int; 
                       lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), isdouble = false, tau = 0.0, usegpu = true) where {E <: AbstractEnv, M <: AbstractValueBasedModel}

    nS, nA = spacedim(env), nactions(env)
    onlinemodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)
    targetmodel = M(nS, nA, opt; hiddendims, lossfn, usegpu)

    return DQNLearner{E, M}(onlinemodel, targetmodel, epochs, env, updatemodelsteps, isdouble, tau)
end

function optimizemodel!(learner::DQNLearner, experiences::B, gamma, step; usegpu = true) where B <: AbstractBuffer 
    optimizemodel!(learner.onlinemodel, experiences, learner.epochs, gamma, argmaxmodel = learner.isdoublelearner ? learner.onlinemodel : learner.targetmodel, targetmodel = learner.targetmodel, usegpu = usegpu)

    if learner.tau > 0
        newparams = [(1 - learner.tau) * targetp + learner.tau * onlinep for (targetp, onlinep) in zip(Flux.params(learner.targetmodel), Flux.params(learner.onlinemodel))]
        Flux.loadparams!(learner.targetmodel, newparams)
    elseif (step % learner.updatemodelsteps) == 0 
        learner.targetmodel = deepcopy(learner.onlinemodel) 
    end
end

(learner::DQNLearner)(state) = learner.onlinemodel(state)
