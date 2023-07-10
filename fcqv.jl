struct FCQV <: AbstractActionValueModel # Fully Connected Q-function
    model::Flux.Chain
    lossfn
end

@functor FCQV (model,)

function FCQV(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 2)

    inputandh1_layer = Flux.PairwiseFusion(vcat, Flux.Dense(inputdims => hiddendims[1], actfn), 
                                                 Flux.Dense(hiddendims[1] + outputdims => hiddendims[2], actfn))

    for i in 2:(length(hiddendims) - 2)
        hiddenlayers[i] = Flux.Dense(hiddendims[i + 1] => hiddendims[i + 2], actfn)
    end

    modelchain = Flux.Chain(inputandh1_layer, last, hiddenlayers..., Flux.Dense(hiddendims[end] => 1))

    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCQV(modelchain, lossfn)
end

ℒ(m::FCQV, v̂, v) = m.lossfn(v̂, v)
𝒬(m::FCQV, state, action) = m.model((state, action)) |> vec
