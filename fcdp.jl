struct FCDP <: AbstractPolicyModel # Fully connected deterministic policy
    model::Flux.Chain
end

@functor FCDP (model,)

function FCDP(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, outactfn = Flux.tanh_fast, usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(Flux.Dense(inputdims => hiddendims[1], actfn), hiddenlayers..., Flux.Dense(hiddendims[end] => outputdims), outactfn)

    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCDP(modelchain)
end

π(m::FCDP, state) = m.model(state) 

function DoubleNetworkActorCriticModel{FCDP, VM}(ninputdims::Int, noutputdims::Int, policyhiddendims::Vector{Int}, valuehiddendims::Vector{Int}, policyopt::Flux.Optimise.AbstractOptimiser, valueopt::Flux.Optimise.AbstractOptimiser;
                                                 usegpu = true) where {VM <: AbstractValueModel}
    policymodel = FCDP(ninputdims, noutputdims, policyhiddendims; usegpu)
    valuemodel = VM(ninputdims, noutputdims, valuehiddendims; usegpu)

    DoubleNetworkActorCriticModel{FCDP, VM}(policymodel, valuemodel, (policymodel = Flux.setup(policyopt, policymodel), valuemodel = Flux.setup(valueopt, valuemodel)))
end