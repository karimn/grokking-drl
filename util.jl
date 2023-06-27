struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

# All implementation of AbstractModel should have a @functor declaration
function update_target_model!(to::M, from::M; τ) where {M <: AbstractModel}
    if τ > 0.0
        newparams = [(1 - τ) * toparam + τ * fromparam for (toparam, fromparam) in zip(Flux.params(to), Flux.params(from))]
        Flux.loadparams!(to, newparams)
    end

    return to 
end

const GaussianPolicy{T, V} = Distributions.MvNormal{T, PDMats.PDiagMat{T, V}, V}

multiple_normals(μ, σ, outputdims) = @pipe vcat(μ, σ) |> 
    Flux.cpu |> 
    eachcol |> 
    reshape.(_, outputdims, :) |> 
    permutedims.(_) |> 
    [[Distributions.Normal(c...) for c in eachcol(datarow)] for datarow in _]

function calcgaes(values, rewards, λ_discounts; N = 1, γ = 1.0)
    T = length(rewards) ÷ N

    advs = reshape(rewards[1:(end - N)] .+ γ * values[(N + 1):end] - values[1:(end - N)], N, :)
    gaes = reduce(hcat, sum(λ_discounts[1:(T - t)]' .* advs[:, t:end], dims = 2) for t in 1:(T - 1)) 

    return advs, gaes 
end

struct NaNParamException <: Base.Exception
    model::AbstractModel
    #prevmodel::AbstractModel
    #Ψ::Vector{Float32}
    states
    #actions
    #discounts
    #values
    #returns
    #β
end

Base.showerror(io::IO, ::NaNParamException) = print(io, "NaNParamException: found NaN values in network parameters")

#(e::NaNParamException)() = e.prevmodel(e.states)

struct CorruptedNetworkException <: Base.Exception 
    prevlearner
    states
    actions
    rewards
    innerex

    CorruptedNetworkException(args...; innerex = nothing) = new(args..., innerex)
end

struct WorkerException <: Base.Exception
    workerid::Int
    sharedmodel::AbstractModel
    localmodel::AbstractModel

    innerexception::Base.Exception
end

Base.showerror(io::IO, e::WorkerException) = print(io, "WorkerException: exception raised in worker $(e.workerid): $(e.innerexception)")

struct NotFiniteLossException <: Base.Exception 
    model::AbstractModel
    states
    actions 
    rewards
    returns
    discounts
    λ_discounts 
end

struct GradientException <: Base.Exception
    model::AbstractModel
    states
    actions
    returns
    innerex
    Ψ
    N
    T
    λ_discounts
    grads
end