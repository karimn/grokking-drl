struct FCGP <: AbstractPolicyModel
    model::Flux.Chain
    outactfn
end

@functor FCGP (model,)

function FCGP(inputdims::Int, outputdims::Int, hiddendims::Vector{Int} = [32, 32]; actfn = Flux.relu, outactfn = Flux.tanh_fast, log_std_bounds = (-20, 2), usegpu = true)
    hiddenlayers = Vector{Any}(nothing, length(hiddendims) - 1)

    for i in 1:(length(hiddendims) - 1)
        hiddenlayers[i] = Flux.Dense(hiddendims[i] => hiddendims[i + 1], actfn)
    end

    modelchain = Flux.Chain(
        Flux.Dense(inputdims => hiddendims[1], actfn), 
        hiddenlayers..., 
        Flux.Parallel(
            (μ, σ) -> (μ, σ),
            mean = Flux.Dense(hiddendims[end] => outputdims), 
            std = Flux.Chain(Flux.Dense(hiddendims[end] => outputdims), o -> clamp.(o, log_std_bounds...), o -> exp.(o))
        )
    )

    if usegpu
        modelchain = modelchain |> Flux.gpu
    end

    return FCGP(modelchain, outactfn)
end

π(m::FCGP, state) = m.model(state)

function Base.rand(rng::Random.AbstractRNG, m::FCGP, state; usegpu = true, ε = 1f-6)
    μ, σ = π(m, usegpu ? Flux.gpu(state) : state)
    z = Flux.ignore(() -> randn(rng, Float32, size(μ)))
    pre_â = μ .+ σ .* (usegpu ? Flux.gpu(z) : z) 
    â = m.outactfn.(pre_â)

    # We need to use RLCore.normlogpdf and not the Distributions.logpdf function because the former is differentiable while the latter 
    # isn't for some reason.
    logπ = sum(RLCore.normlogpdf(μ, σ, pre_â) .- log.(clamp.(1 .- â.^2, 0, 1) .+ ε), dims = 1) 

    # This is what the Spinning Up code does in https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L53
    # Apparently, it is more numerically stable, but I don't know how it is derived.
    #logπ = sum(RLCore.normlogpdf(μ, σ, â) .- (2.0f0 .* (log(2.0f0) .- â .- RLCore.softplus.(-2.0f0 .* â))), dims=1)

    return Flux.cpu(â), Flux.cpu(logπ)
end



selectaction(m::FCGP, state; rng::Random.AbstractRNG = Random.GLOBAL_RNG, use_gpu = true) = Base.rand(rng, m, usegpu ? Flux.gpu(state) : state) |> vec
selectgreedyaction(m::FCGP, state; usegpu = true) = @pipe π(m, usegpu ? Flux.gpu(state) : state) |> first |> mean.(_) |> m.outactfn.(_) |> vec