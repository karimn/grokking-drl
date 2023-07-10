function CreateSimpleFCModel(::Type{M}, inputdim::Int, outputdim::Int, opt::Union{Nothing, Flux.Optimise.AbstractOptimiser} = nothing; hiddendims::Vector{Int}, actfn, lossfn, usegpu) where M <: AbstractModel
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdim => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Dense(hiddendims[end] => outputdim)
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    if opt ≢ nothing
        opt = Flux.setup(opt, modelchain)

        return M(modelchain, opt, lossfn)
    else
        return M(modelchain, lossfn)
    end
end
