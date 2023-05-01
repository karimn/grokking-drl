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

function optimizemodel!(onlinemodel::FCQ, experiences::B, epochs, gamma; targetmodel::FCQ = onlinemodel, argmaxmodel = targetmodel, usegpu = true) where B <: AbstractBuffer
    batch = getbatch(experiences)

    actions = batch.a # [e.a for e in experiences]

    for _ in epochs
        sp = @pipe mapreduce(permutedims, vcat, batch.sp) |>
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) 

        argmax_a_q_sp = @pipe argmaxmodel(sp) |>
            argmax(_, dims = 1) |> 
            Flux.cpu

        q_sp = targetmodel(sp) 
        max_a_q_sp = q_sp[argmax_a_q_sp] |> Flux.cpu 
        target_q_s = @pipe [r + gamma * q * (!failure) for (r, q, failure) in zip(batch.r, max_a_q_sp, batch.failure)]  

        @pipe mapreduce(permutedims, vcat, batch.s) |> 
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) |> 
            (_, target_q_s) |> 
            train!(Flux.mse, onlinemodel, _, actions)  
    end
end

function save(m::FCQ, filename) 
    model = m.model |> Flux.cpu
    BSON.@save filename model
end