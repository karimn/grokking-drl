struct FCCA <: AbstractPolicyModel
    model::Flux.Chain
end

@functor FCCA (model,)

function FCCA(inputdim::Int, outputdim::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, lossfn = (ŷ, y, args...) -> Flux.mse(ŷ, y), usegpu = true)
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

    return FCCA(modelchain)
end

π(m::FCCA, state) = m.model(state)
π(m::FCCA, state::Vector{Vector}) = reduce(hcat, state) |> m.model

function Base.rand(rng::Random.AbstractRNG, m::FCCA, state) 
    dist = @pipe π(m, state) |> 
        Flux.cpu |> 
        Distributions.Categorical.(copy(p) for p in eachcol(_))

    actions = rand.(rng, dist)
    logπ = Distributions.logpdf.(dist, actions)

   return actions, logπ 
end

Base.rand(m::FCCA, state) = rand(Random.GLOBAL_RNG, m, state)

function get_predictions(m::FCCA, state, action)
    #=dist = @pipe π(m, state) |> 
        Flux.cpu |> 
        Distributions.Categorical.(copy(p) for p in eachcol(_))

    Distributions.Categorical.(copy(p) for p in eachcol(_))

    return Distributions.logpdf.(dist, action), Distributions.entropy.(dist)
    =#

    # I have profiled this, but the below code seems faster. Not sure how numerically stable it is.

    pmfmat = π(m, state) |> Flux.cpu 
    lpmfmat = log.(pmfmat .+ 1f-6)

    return [lp[a] for (lp, a) in zip(eachcol(lpmfmat), action)], -vec(sum(pmfmat .* lpmfmat, dims = 1)) 
end

save(m::FCCA, filename; args...) = JLD2.jldsave(filename; m = Flux.state(m), args...)