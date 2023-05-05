struct FCDuelingQ <: AbstractValueModel
    model
    opt
    lossfn
end

(m::FCDuelingQ)(state) = m.model(state) 

function FCDuelingQ(inputdim::Int, outputdim::Int, valueopt::Flux.Optimise.AbstractOptimiser; hiddendims::Vector{Int} = [32, 32], actfn = Flux.relu, lossfn = (ŷ, y, w) -> Flux.mse(ŷ, y), usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdim => hiddendims[1], actfn), 
        hiddenlayers...,
        Flux.Parallel(Flux.Dense(hiddendims[end] => outputdim), Flux.Dense(hiddendims[end] => 1)) do adv, val 
            val .+ adv .- Statistics.mean(adv)
        end
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    opt = Flux.setup(valueopt, modelchain)

    return FCDuelingQ(modelchain, opt, lossfn)
end