struct FCTQV <: AbstractValueModel
    model₁::FCQSA
    model₂::FCQSA
end

@functor FCTQV (model₁, model₂)

function FCTQV(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return FCTQV(FCQSA(inputdims, outputdims, hiddendims; actfn, lossfn, usegpu), FCQSA(inputdims, outputdims, hiddendims; actfn, lossfn, usegpu))
end

ℒ(m::FCTQV, v̂₁, v̂₂, v) = ℒ(m.model₁, v̂₁, v) + ℒ(m.model₂, v̂₂, v)
𝒬₁(m::FCTQV, state, action) = 𝒬(m.model₁, state, action)
𝒬₂(m::FCTQV, state, action) = 𝒬(m.model₂, state, action)
𝒬(m::FCTQV, state, action) = 𝒬₁(m, state, action), 𝒬₂(m, state, action) 