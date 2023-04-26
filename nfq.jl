struct FCQ <: AbstractModel
    model
    opt
end

function FCQ(inputdim::Int, outputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdim => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Dense(hiddendims[end] => outputdim))
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    opt = Flux.setup(valueopt, modelchain)

    return FCQ(modelchain, opt)
end

(m::FCQ)(state) = m.model(state) 

function train!(loss, m::FCQ, data, actions) 
    #Flux.train!(loss, m.model, data, m.opt) 
    
    input, label = data 

    val, grads = Flux.withgradient(m.model) do m
        fullresult = m(input) |> Flux.cpu
        result = [r[a] for (r, a) in zip(eachcol(fullresult), actions)]
        loss(result, label)
    end

    if !isfinite(val)
        @warn "loss is $val"
    end

    Flux.update!(m.opt, m.model, grads[1])
end


mutable struct NFQ{M <: AbstractModel}
    hiddendims::Vector{Int}
    valueopt::Flux.Optimise.AbstractOptimiser
    trainstrategy::AbstractStrategy
    evalstrategy::AbstractStrategy
    batchsize::Int
    epochs::Int
    onlinemodel::Union{Nothing, M}

    NFQ{M}(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs) where M = new{M}(hiddendims, valueopt, trainstrategy, evalstrategy, batchsize, epochs, nothing) 
end

function train!(agent::NFQ{M}, env::AbstractEnv, gamma::Float64, maxminutes::Int, maxepisodes::Int; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where M
    nS, nA = spacedim(env), nactions(env)

    agent.onlinemodel = M(nS, nA, agent.valueopt, hiddendims = agent.hiddendims, usegpu = usegpu) 

    episodereward = Float64[]
    episodetimestep = Int[]
    experiences = []

    for _ in 1:maxepisodes
        reset!(env)

        currstate, isterminal = state(env), is_terminated(env)
        push!(episodereward, 0)
        push!(episodetimestep, 0)

        step = 1

        while true
            # Interaction step
            action = selectaction(agent.trainstrategy, agent.onlinemodel, currstate, rng = rng, usegpu = usegpu)
            env(action)
            newstate = state(env)
            episodereward[end] += curr_reward = reward(env)
            episodetimestep[end] += 1
            isterminal = is_terminated(env)
            istruncated = false # is_truncated(env)

            push!(experiences, (s = currstate, a = action, r = curr_reward, sp = newstate, failure = isterminal && !istruncated))
            currstate = newstate

            if length(experiences) >= agent.batchsize
                actions = [e.a for e in experiences]

                for _ in agent.epochs
                    max_a_q_sp = @pipe mapreduce(permutedims, vcat, [e.sp for e in experiences]) |>
                        permutedims |>
                        (usegpu ? Flux.gpu(_) : _) |> 
                        agent.onlinemodel |>
                        maximum(_, dims = 1) |> 
                        Flux.cpu

                    target_q_s = [e.r + gamma * q * (!e.failure) for (e, q) in zip(experiences, max_a_q_sp)]

                    @pipe mapreduce(permutedims, vcat, [e.s for e in experiences]) |>
                        permutedims |>
                        (usegpu ? Flux.gpu(_) : _) |> 
                        (_, target_q_s) |> 
                        train!(Flux.mse, agent.onlinemodel, _, actions)  
                        #  train!(agent.onlinemodel, _) do m, x, y
                        #      results = m(x)

                        #      Flux.mse([r[e.a] for (r, e) in zip(eachcol(results), experiences)], target_q_s)
                        #  end |> 
                        #(usegpu ? Flux.cpu(_) : _)
                end

               empty!(experiences) 
            end

            if isterminal
                break
            else
                step += 1
            end
        end
    end

    evaluate_model(agent.evalstrategy, agent.onlinemodel, env::AbstractEnv, nepisodes = 100, usegpu = usegpu)
end