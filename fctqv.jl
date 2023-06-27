struct FCTQV <: AbstractValueModel
    model₁::FCQSA
    model₂::FCQSA
end

@functor FCTQV (model₁, model₂)

function FCTQV(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return FCTQV(FCQSA(inputdims, outputdims, hiddendims; actfn, lossfn, usegpu), FCQSA(inputdims, outputdims, hiddendims; actfn, lossfn, usegpu))
end

𝒬₁(m::FCTQV, state, action) = 𝒬(m.model₁, state, action)
𝒬₂(m::FCTQV, state, action) = 𝒬(m.model₂, state, action)
𝒬(m::FCTQV, state, action) = 𝒬₁(m, state, action), 𝒬₂(m, state, action) 
min𝒬(m::FCTQV, state, action) = min.(vec.(Flux.cpu(𝒬(m, state, action)))...)
ℒ₁(m::FCTQV, s, a, v) = ℒ(m.model₁, vec(𝒬₁(m, s, a)), v)
ℒ₂(m::FCTQV, s, a, v) = ℒ(m.model₂, vec(𝒬₂(m, s, a)), v)
ℒ₁(m::FCTQV, v̂, v) = ℒ(m.model₁, v̂, v)
ℒ₂(m::FCTQV, v̂, v) = ℒ(m.model₂, v̂, v)
ℒ(m::FCTQV, v̂₁, v̂₂, v) = ℒ₁(m, v̂₁, v) + ℒ₂(m, v̂₂, v)
ℒ(m::FCTQV, s, a, v) = ℒ₁(m, s, a, v) + ℒ₂(m, s, a, v)