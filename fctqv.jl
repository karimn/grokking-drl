struct FCTQV <: AbstractValueModel
    model₁::Flux.Chain
    model₂::Flux.Chain
    lossfn
end

@functor FCTQV (model₁, model₂)

function FCTQV(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    hiddenlayers₁ = Vector{Any}(nothing, length(hiddendims) - 1)
    hiddenlayers₂ = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers₁[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
        hiddenlayers₂[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain₁ = Flux.Chain(
        Flux.Dense(inputdims + outputdims => hiddendims[1], actfn), 
        hiddenlayers₁..., 
        Flux.Dense(hiddendims[end] => 1)
    )

    modelchain₂ = Flux.Chain(
        Flux.Dense(inputdims + outputdims => hiddendims[1], actfn), 
        hiddenlayers₂..., 
        Flux.Dense(hiddendims[end] => 1)
    )
        
    if usegpu
        modelchain₁ = modelchain₁ |> Flux.gpu
        modelchain₂ = modelchain₂ |> Flux.gpu
    end

    return FCTQV(modelchain₁, modelchain₂, lossfn)
end

ℒ(m::FCTQV, v̂₁, v̂₂, v) = m.lossfn(v̂₁, v) + m.lossfn(v̂₂, v)
𝒬₁(m::FCTQV, state, action) = m.model₁(vcat(state, action))
𝒬₂(m::FCTQV, state, action) = m.model₂(vcat(state, action))
𝒬(m::FCTQV, state, action) = 𝒬₁(m, state, action), 𝒬₂(m, state, action) 