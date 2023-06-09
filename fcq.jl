# Fully connected Q-function
struct FCQ <: AbstractValueModel
    model
    opt
    lossfn
end

function FCQ(inputdim::Int, outputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return CreateSimpleFCModel(FCQ, inputdim, outputdim, valueopt; hiddendims, actfn, lossfn, usegpu)
end

(m::FCQ)(state) = m.model(state) 

function train!(m::M, data, actions, weights) where M <: AbstractValueModel 
    input, label = data 
    local tderrors

    val, grads = Flux.withgradient(m.model) do modelchain
        fullresult = modelchain(input) |> Flux.cpu
        result = [r[a] for (r, a) in zip(eachcol(fullresult), actions)]
        tderrors = Vector{Float32}(result - label)

        (m.lossfn)(result, label, weights)
    end

    if !isfinite(val)
        @warn "loss is $val"
    end

    Flux.update!(m.opt, m.model, grads[1])

    return tderrors
end

function optimizemodel!(onlinemodel::M, experiences::B, epochs, γ; targetmodel::M = onlinemodel, argmaxmodel::M = targetmodel, usegpu = true) where {B <: AbstractBuffer, M <: AbstractValueModel}
    for _ in epochs
        idxs, weights, batch = getbatch(experiences)
        actions = batch.a 

        sp = @pipe mapreduce(permutedims, vcat, batch.sp) |>
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) 

        argmax_a_q_sp = @pipe argmaxmodel(sp) |>
            argmax(_, dims = 1) |> 
            Flux.cpu

        q_sp = targetmodel(sp) 
        max_a_q_sp = q_sp[argmax_a_q_sp] |> Flux.cpu 
        target_q_s = @pipe [r + γ * q * (!failure) for (r, q, failure) in zip(batch.r, max_a_q_sp, batch.failure)]  

        @pipe mapreduce(permutedims, vcat, batch.s) |> 
            permutedims |>
            (usegpu ? Flux.gpu(_) : _) |> 
            (_, target_q_s) |> 
            train!(onlinemodel, _, actions, weights) |> 
            update!(experiences, idxs, _)
    end
end

opt(m::FCQ) = m.opt

function save(m::M, filename) where M <: AbstractValueModel
    model = m.model |> Flux.cpu
    BSON.@save filename model
end