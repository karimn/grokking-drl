# Fully connected Q-function
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

function optimizemodel!(onlinemodel::FCQ, experiences, epochs, gamma; targetmodel::FCQ = onlinemodel, usegpu = true) 
    actions = [e.a for e in experiences]

    for _ in epochs
        max_a_q_sp = @pipe mapreduce(permutedims, vcat, [e.sp for e in experiences]) |>
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) |> 
            targetmodel |>
            maximum(_, dims = 1) |> 
            Flux.cpu

        target_q_s = [e.r + gamma * q * (!e.failure) for (e, q) in zip(experiences, max_a_q_sp)]

        @pipe mapreduce(permutedims, vcat, [e.s for e in experiences]) |>
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) |> 
            (_, target_q_s) |> 
            train!(Flux.mse, onlinemodel, _, actions)  
            #  train!(agent.onlinemodel, _) do m, x, y
            #      results = m(x)

            #      Flux.mse([r[e.a] for (r, e) in zip(eachcol(results), experiences)], target_q_s)
            #  end |> 
            #(usegpu ? Flux.cpu(_) : _)
    end
end
