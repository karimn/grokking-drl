struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

function update!(to::Flux.Chain, from::Flux.Chain; τ = 1.0)
    if τ > 0.0
        newparams = [(1 - τ) * toparam + τ * fromparam for (toparam, fromparam) in zip(Flux.params(to), Flux.params(from))]
        Flux.loadparams!(to, newparams)
    end

    return to 
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
end