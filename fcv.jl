struct FCV <: AbstractValueModel
    model
    lossfn
end

@functor FCV (model,) 

𝒱(m::FCV, state) = m.model(state)
ℒ(m::FCV, v̂, v) = m.lossfn(v̂, v)

function FCV(inputdim::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
    return CreateSimpleFCModel(FCV, inputdim, 1; hiddendims, actfn, lossfn, usegpu)
end

