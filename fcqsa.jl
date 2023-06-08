struct FCQSA <: AbstractValueModel
    model::Flux.Chain
    lossfn
end

@functor FCQSA (model,) 

function FCQSA(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdims + outputdims => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Dense(hiddendims[end] => 1)
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCQSA(modelchain, lossfn)
end

ℒ(m::FCQSA, v̂, v) = m.lossfn(v̂, v)
𝒬(m::FCQSA, state, action) = m.model(vcat(state, action))

