struct FCDAP <: AbstractPolicyModel
    model
end

@functor FCDAP

function FCDAP(inputdim::Int, outputdim::Int, hiddendims::Vector{Int}; actfn = Flux.relu, usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdim => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Dense(hiddendims[end] => outputdim),
        Flux.softmax
    )
        
    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCDAP(modelchain)
end

π(m::FCDAP, state) = m.model(state)

function train!(m::FCDAP, states, actions, rewards, opt; γ = 1.0)
    T = size(states, 2)
    discounts = γ.^range(0, T - 1)
    returns = [sum(discounts[begin:(T - t + 1)] .* rewards[t:end]) for t in 1:T] 

    val, grads = Flux.withgradient(m, states, actions, returns, discounts) do policymodel, s, a, r, d
        lpdf = @pipe π(policymodel, s) |> 
            log.(_) |> 
            Flux.cpu |> 
            [collpdf[colact] for (collpdf, colact) in zip(eachcol(_), a)] 

        - mean(d .* r .* lpdf)
    end

    !isfinite(val) && @warn "loss is $val"

    Flux.update!(opt, m, grads[1])
end
