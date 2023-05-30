const DDPGModel = DoubleNetworkActorCriticModel{FCDP, FCQV}

𝒬(m::DDPGModel, state, action) = 𝒬(m.valuemodel, state, action)
#𝒬(m::DDPGModel, state, action::Vector) = 𝒬(m.valuemodel, state, permutedims(action))
π(m::DDPGModel, state) = π(m.policymodel, state)

struct DDPGLearner{E} <: AbstractValueLearner where {E <: AbstractEnv}
    targetmodel::DDPGModel
    onlinemodel::DDPGModel
    env::E
    updatemodelsteps::Int
    τ::Float32
end

function DDPGLearner(env::E, dnargs...; updatemodelsteps, τ = 1.0, usegpu = true) where {E <: AbstractEnv} 
    nS, nA = spacedim(env), nactions(env)
    #actionbounds = Float32[minimum(action_space(env)) maximum(action_space(env))]
    targetmodel = DDPGModel(nS, nA, dnargs...; usegpu)
    onlinemodel = DDPGModel(nS, nA, dnargs...; usegpu)

    update_target_model!(targetmodel, onlinemodel)

    return DDPGLearner{E}(targetmodel, onlinemodel, env, updatemodelsteps, τ) 
end

#=
function step!(learner::DDPGLearner, s::AbstractStrategy, currstate; rng = Random.GLOBAL_RNG, usegpu = true)
    action, _ = selectaction!(s, learner.onlinemodel, currstate; rng, usegpu)
    learner.env(action)
    newstate = Vector{Float32}(state(learner.env))

    return action, newstate, reward(learner.env), is_terminated(learner.env), istruncated(learner.env) 

end

evaluate(evalstrategy::AbstractStrategy, learner::DDPGLearner, env::E; nepisodes = 1, usegpu = true) where {E <: AbstractEnv} = evaluate(evalstrategy, learner.onlinemodel, env, nepisodes = nepisodes, usegpu = usegpu)
=#

function optimizemodel!(learner::DDPGLearner, experiences::AbstractBuffer, γ, step; usegpu = true)
    _..., batch = getbatch(experiences)
    actions = (usegpu ? Flux.gpu(batch.a) : batch.a) |> permutedims 

    s = @pipe reduce(hcat, batch.s) |> (usegpu ? Flux.gpu(_) : _)
    sp = @pipe reduce(hcat, batch.sp) |> (usegpu ? Flux.gpu(_) : _) 

    argmax_a_q_sp = π(learner.targetmodel, sp)
    max_a_q_sp = 𝒬(learner.targetmodel, sp, argmax_a_q_sp) #|> Flux.cpu 

    if usegpu
        target_q_sa = Flux.gpu(batch.r) + γ * max_a_q_sp .* Flux.gpu(.!batch.failure)
    else
        target_q_sa = batch.r + γ * max_a_q_sp .* (.!batch.failure)
    end

    #vval, vgrads = Flux.withgradient(learner.onlinemodel) do m
    vval, vgrads = Flux.withgradient(m -> ℒ(m, 𝒬(m, s, actions), target_q_sa), learner.onlinemodel.valuemodel) 

    isfinite(vval) || @warn "Value loss is $vval"

    Flux.update!(learner.onlinemodel.opt.valuemodel, learner.onlinemodel.valuemodel, vgrads[1])
    newvval, _= Flux.withgradient(m -> ℒ(m, 𝒬(m, s, actions), target_q_sa), learner.onlinemodel.valuemodel) 

    #pval, pgrads = Flux.withgradient(learner.onlinemodel) do m
    #    - mean(𝒬(m, s, π(m, s)))
    #end

    pval, pgrads = Flux.withgradient((pm, vm) -> - mean(𝒬(vm, s, π(pm, s))), learner.onlinemodel.policymodel, learner.onlinemodel.valuemodel) 

    #@debug "Loss" minimum(argmax_a_q_sp) maximum(argmax_a_q_sp) vval pval

    isfinite(pval) || @warn "Policy loss is $pval"

    #Flux.update!(learner.onlinemodel, vgrads[1], pgrads[1])

    Flux.update!(learner.onlinemodel.opt.policymodel, learner.onlinemodel.policymodel, pgrads[1])
    newpval, _ = Flux.withgradient((pm, vm) -> - mean(𝒬(vm, s, π(pm, s))), learner.onlinemodel.policymodel, learner.onlinemodel.valuemodel) 

    #@debug "Updated loss" newvval newpval

    (step % learner.updatemodelsteps) == 0 && update_target_model!(learner)
end

#=
function Flux.train!(m::DDPGModel, states, actions, target_q_sa; γ = 1.0)
    vval, vgrads = Flux.withgradient(m.valuemodel) do vm
        q_sa = 𝒬(vm, states, actions)

        ℒ(m, q_sa, target_q_sa)
    end

    isfinite(vval) || @warn "Value loss is $vval"

    pval, pgrads = Flux.withgradient(m) do dm
        - mean(𝒬(dm, states, π(dm, states)))
    end

    isfinite(pval) || @warn "Policy loss is $pval"

    Flux.update!(m, pgrads, vgrads)
end
=#