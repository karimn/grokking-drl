const TD3Model = DoubleNetworkActorCriticModel{FCDP, FCTQV}

ℒᵥ(m::TD3Model, v̂₁, v̂₂, v) = ℒ(m.valuemodel, v̂₁, v̂₂, v)

struct TD3Learner{E} <: AbstractActorCriticLearner where {E <: AbstractEnv} 
    targetmodel::TD3Model
    onlinemodel::TD3Model
    env::E
    experiences::ReplayBuffer
    update_policy_model_steps::Int
    update_value_model_steps::Int
    train_policy_model_steps::Int
    policy_noise_ratio::Float32
    policy_noise_clip_ratio::Float32
    τ::Float32
end

environment(l::TD3Learner) = l.env
buffer(l::TD3Learner) = l.experiences
readybatch(l::TD3Learner) = readybatch(l.experiences)

function TD3Learner(env::E, experiences::ReplayBuffer, dnargs...; 
                    update_policy_model_steps::Int, update_value_model_steps::Int, train_policy_model_steps, policy_noise_ratio::Float32, policy_noise_clip_ratio::Float32, τ = 1.0, usegpu = true) where E <: AbstractEnv
    nS, nA = spacedim(env), nactions(env)
    targetmodel = TD3Model(nS, nA, dnargs...; usegpu)
    onlinemodel = TD3Model(nS, nA, dnargs...; usegpu)

    update_target_model!(targetmodel, onlinemodel, τ = 1.0)

    return TD3Learner{E}(targetmodel, onlinemodel, env, experiences, update_policy_model_steps, update_value_model_steps, train_policy_model_steps, policy_noise_ratio, policy_noise_clip_ratio, τ) 
end

function optimizemodel!(learner::TD3Learner, experiences::AbstractBuffer, γ, totalsteps; usegpu = true)
    _..., batch = getbatch(experiences)
    actions = reduce(hcat, usegpu ? Flux.gpu(batch.a) : batch.a)  

    s = @pipe reduce(hcat, batch.s) |> (usegpu ? Flux.gpu(_) : _)
    sp = @pipe reduce(hcat, batch.sp) |> (usegpu ? Flux.gpu(_) : _) 

    a_noise = @pipe (rand(Distributions.Normal(), size(actions)...) * learner.policy_noise_ratio * 2) |>  # The range is always from -1 to 1
        (usegpu ? Flux.gpu(_) : _)
    clamp!(a_noise, -learner.policy_noise_clip_ratio, learner.policy_noise_clip_ratio)

    argmax_a_q_sp = π(learner.targetmodel, sp)
    noisy_argmax_a_q_sp = clamp.(argmax_a_q_sp .+ a_noise, -1, 1)

    max_a_q_sp₁, max_a_q_sp₂ = vec.(𝒬(learner.targetmodel, sp, noisy_argmax_a_q_sp))
    @CUDA.allowscalar max_a_q_sp = min(max_a_q_sp₁, max_a_q_sp₂)

    if usegpu
        target_q_sa = Flux.gpu(batch.r) + γ * max_a_q_sp .* Flux.gpu(.!batch.failure)
    else
        target_q_sa = batch.r + γ * max_a_q_sp .* (.!batch.failure)
    end

    vval, vgrads = Flux.withgradient(m -> ℒᵥ(m, vec.(𝒬(m, s, actions))..., target_q_sa), learner.onlinemodel) 

    isfinite(vval) || @warn "Value loss is $vval"

    #save(learner.onlinemodel, "temp-data/td3_checkpoint.jld2"; vgrads)

    Flux.update!(learner.onlinemodel, vgrads[1])

    if totalsteps % learner.train_policy_model_steps == 0

        # It's important to get gradients for and update the parameters of the policy model *only*. Don't use learner.onlinemodel as above. I think that makes us update
        # both the policy and value models.
        pval, pgrads = Flux.withgradient((pm, vm) -> - mean(𝒬₁(vm, s, π(pm, s))), learner.onlinemodel.policymodel, learner.onlinemodel.valuemodel) 

        isfinite(pval) || @warn "Policy loss is $pval"

        #save(learner.onlinemodel, "temp-data/td3_checkpoint.jld2"; vgrads, pgrads)

        Flux.update!(learner.onlinemodel.opt.policymodel, learner.onlinemodel.policymodel, pgrads[1])
    end

    (totalsteps % learner.update_policy_model_steps) == 0 && update_target_policy_model!(learner)
    (totalsteps % learner.update_value_model_steps) == 0 && update_target_value_model!(learner)
end

update_target_policy_model!(l::TD3Learner) = update_target_policy_model!(l.targetmodel, l.onlinemodel, τ = l.τ)
update_target_value_model!(l::TD3Learner) = update_target_value_model!(l.targetmodel, l.onlinemodel, τ = l.τ)