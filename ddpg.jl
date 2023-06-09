const DDPGModel = DoubleNetworkActorCriticModel{FCDP, FCQV}

struct DDPGLearner{E} <: AbstractActorCriticLearner where {E <: AbstractEnv}
    targetmodel::DDPGModel
    onlinemodel::DDPGModel
    env::E
    experiences::ReplayBuffer
    updatemodelsteps::Int
    τ::Float32
end

environment(l::DDPGLearner) = l.env
buffer(l::DDPGLearner) = l.experiences
readybatch(l::DDPGLearner) = readybatch(l.experiences)

function DDPGLearner(env::E, experiences::ReplayBuffer, dnargs...; updatemodelsteps, τ = 1.0, usegpu = true) where {E <: AbstractEnv} 
    nS, nA = spacedim(env), nactions(env)
    targetmodel = DDPGModel(nS, nA, dnargs...; usegpu)
    onlinemodel = DDPGModel(nS, nA, dnargs...; usegpu)

    update_target_model!(targetmodel, onlinemodel, τ = 1.0)

    return DDPGLearner{E}(targetmodel, onlinemodel, env, experiences, updatemodelsteps, τ) 
end

function optimizemodel!(learner::DDPGLearner, experiences::AbstractBuffer, γ, step; usegpu = true)
    _..., batch = getbatch(experiences)
    actions = reduce(hcat, usegpu ? Flux.gpu(batch.a) : batch.a)  

    s = @pipe reduce(hcat, batch.s) |> (usegpu ? Flux.gpu(_) : _)
    sp = @pipe reduce(hcat, batch.sp) |> (usegpu ? Flux.gpu(_) : _) 

    argmax_a_q_sp = π(learner.targetmodel, sp)
    max_a_q_sp = 𝒬(learner.targetmodel, sp, argmax_a_q_sp)

    if usegpu
        target_q_sa = Flux.gpu(batch.r) + γ * max_a_q_sp .* Flux.gpu(.!batch.failure)
    else
        target_q_sa = batch.r + γ * max_a_q_sp .* (.!batch.failure)
    end

    vval, vgrads = Flux.withgradient(m -> ℒᵥ(m, 𝒬(m, s, actions), target_q_sa), learner.onlinemodel) 

    isfinite(vval) || @warn "Value loss is $vval"

    Flux.update!(learner.onlinemodel, vgrads[1])

    # It's important to get gradients for and update the parameters of the policy model *only*. Don't use learner.onlinemodel as above. I think that makes us update
    # both the policy and value models.
    pval, pgrads = Flux.withgradient((pm, vm) -> - mean(𝒬(vm, s, π(pm, s))), learner.onlinemodel.policymodel, learner.onlinemodel.valuemodel) 

    isfinite(pval) || @warn "Policy loss is $pval"

    Flux.update!(learner.onlinemodel.opt.policymodel, learner.onlinemodel.policymodel, pgrads[1])

    (step % learner.updatemodelsteps) == 0 && update_target_model!(learner)
end
