struct PPOLearner <: AbstractActorCriticLearner
    policymodel::FCCA
    valuemodel::FCV
    policymodelopt
    valuemodelopt
    policy_sample_ratio::Float32
    policy_clip_range::Float32
    policy_stop_kl::Float32
    policy_optim_epochs::Int
    value_sample_ratio::Float32
    value_clip_range::Float32
    value_stop_mse::Float32
    value_optim_epochs::Int
    entropylossweight::Float32
    buffer::EpisodeBuffer

    function PPOLearner(buffer::EpisodeBuffer, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser;
                        policy_sample_ratio::Float32, policy_clip_range::Float32, policy_stop_kl::Float32, policy_optim_epochs::Int, 
                        value_sample_ratio::Float32, value_clip_range::Float32, value_stop_mse::Float32, value_optim_epochs::Int, 
                        entropylossweight::Float32, usegpu = true) 
        nS, nA = spacedim(environment(buffer)), nactions(environment(buffer))
        policymodel = FCCA(nS, nA, policyhiddendims; usegpu)
        valuemodel = FCV(nS, valuehiddendims; usegpu)

        return new(policymodel, valuemodel, Flux.setup(policyopt, policymodel), Flux.setup(valueopt, valuemodel), policy_sample_ratio, policy_clip_range, policy_stop_kl, policy_optim_epochs, 
                   value_sample_ratio, value_clip_range, value_stop_mse, value_optim_epochs, entropylossweight, buffer)
    end
end

function train!(learner::PPOLearner; maxminutes::Int, maxepisodes::Int, goal_mean_reward, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    evalscores = []
    episodereward = Float32[]
    episodetimestep = Int[]
    episodeexploration = Int[]
    results = EpisodeResult[]

    trainstart = now()
    episode = 0
    training_is_over = false

    while !training_is_over 
        curr_episode_time_step, curr_episode_reward = fill!(learner.buffer, learner.policymodel, learner.valuemodel; usegpu)
        n_ep_batch = length(curr_episode_time_step)
        append!(episodetimestep, curr_episode_time_step)
        append!(episodereward, curr_episode_reward)

        optimizemodel!(learner; usegpu)

        episode += n_ep_batch

        evalscore, evalscoresd, evalsteps = evaluate(learner.policymodel, environment(learner.buffer); rng, usegpu)
        push!(evalscores, evalscore)     

        mean100evalscores = mean(last(evalscores, 100))

        push!(results, EpisodeResult(sum(length, curr_episode_time_step), mean(last(episodereward, 100)), mean100evalscores)) 

        @debug "Episode completed" episode evalscore evalscoresd evalsteps mean100evalscores 

        rewardgoalreached = mean100evalscores >= goal_mean_reward
        rewardgoalreached && @info "Reached goal mean reward" goal_mean_reward

        reachedmaxepisodes = episode + buffer_maxepisodes(learner.buffer) >= maxepisodes
        reachedmaxepisodes && @info "Reached max episodes" episode 

        wallclockelapsed = now() - trainstart
        maxtimereached = (wallclockelapsed.value / 60_000) >= maxminutes 
        maxtimereached && @info "Max time reached" wallclockelapsed 

        training_is_over = rewardgoalreached || reachedmaxepisodes || maxtimereached

        training_is_over || clear!(learner.buffer)
        usegpu && CUDA.reclaim()
    end

    return results, evaluate(learner.policymodel, environment(learner.buffer); nepisodes = 100, rng, usegpu)
end

function ppo_policy_loss(m::AbstractPolicyModel, states, actions, oldlogπ, gaes, clip_range, entropylossweight)
    newlogπ, ent = get_predictions(m, states, actions)
    ratios = exp.(newlogπ .- oldlogπ) 
    πobj = ratios .* gaes 
    πobj_clipped = gaes .* clamp.(ratios, 1f0 - clip_range, 1f0 + clip_range)

    policyloss = - mean(min.(πobj, πobj_clipped)) 
    entropyloss = - mean(ent) * entropylossweight

    return policyloss + entropyloss
end

function optimizemodel!(learner::PPOLearner; usegpu = true)
    states, actions, returns, logπ, gaes = getstacks(learner.buffer)
    states = usegpu ? Flux.gpu(states) : states
    gaes_mean, gaes_std = @pipe gaes |> (mean(_), std(_))

    gaes = @. (gaes - gaes_mean) / (gaes_std + 1f-6)
    values = 𝒱(learner.valuemodel, states) |> vec

    ns = nsamples(learner.buffer)

    for pepoch in 1:learner.policy_optim_epochs
        batchsize = round(Int, learner.policy_sample_ratio * ns)
        batchidx = sample(1:nsamples(learner.buffer), batchsize, replace = false)
        batchstates = states[:, batchidx]
        batchactions = actions[batchidx]
        batchlogπ = logπ[batchidx]
        batchgaes = gaes[batchidx]

        pval, pgrads = Flux.withgradient(ppo_policy_loss, learner.policymodel, batchstates, batchactions, batchlogπ, batchgaes, learner.policy_clip_range, learner.entropylossweight) 
        # do pm

        # pval, pgrads = Flux.withgradient(learner.policymodel) do pm
        #     newlogπ, ent = get_predictions(pm, batchstates, batchactions)
        #     ratios = exp.(newlogπ .- batchlogπ) 
        #     πobj = ratios .* batchgaes 
        #     πobj_clipped = batchgaes .* clamp.(ratios, 1f0 - learner.policy_clip_range, 1f0 + learner.policy_clip_range)

        #     policyloss = - mean(min.(πobj, πobj_clipped)) 
        #     entropyloss = - mean(ent) * learner.entropylossweight

        #     return policyloss + entropyloss
        # end

        isfinite(pval) || @warn "policy loss is $pval"
        isfinite(pval) || throw(NotFiniteLossException(learner.policymodel, batchstates, batchactions, batchlogπ, batchgaes))

        Flux.update!(learner.policymodelopt, learner.policymodel, pgrads[1])

        policykl = mean(logπ - get_predictions(learner.policymodel, states, actions)[1])

        if policykl > learner.policy_stop_kl 
            @debug "Policy optimization terminated because of high KL divergence." pepoch policykl
            break
        end
    end

    for vepoch in 1:learner.value_optim_epochs
        batchsize = round(Int, learner.value_sample_ratio * ns)
        batchidx = sample(1:nsamples(learner.buffer), batchsize, replace = false) 
        batchstates = states[:, batchidx]
        batchreturns = returns[batchidx] |> Flux.cpu 
        batchvalues = values[batchidx] |> Flux.cpu

        vval, vgrads = Flux.withgradient(learner.valuemodel) do vm
            values_pred = 𝒱(vm, batchstates) |> Flux.cpu |> vec
            values_pred_clipped = batchvalues + clamp.(values_pred - batchvalues, -learner.value_clip_range, learner.value_clip_range)

            vloss = (batchreturns - values_pred).^2
            vloss_clipped = (batchreturns - values_pred_clipped).^2

            return max.(vloss, vloss_clipped) |> mean 
        end

        isfinite(vval) || @warn "value loss is $vval"

        Flux.update!(learner.valuemodelopt, learner.valuemodel, vgrads[1])

        valuemse = Flux.mse(𝒱(learner.valuemodel, states) |> vec, values, agg = x -> mean(x) * 0.5)

        if valuemse > learner.value_stop_mse 
            @debug "Value optimization terminated because of high MSE" vepoch valuemse
            break
        end
    end
end
