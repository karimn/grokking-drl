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

function policyloss(modelchain, s, a, d, Ψ; β) 
    pdist = modelchain(s) |> Flux.cpu 
    ent_lpdf = hcat([[Distributions.entropy(coldist), log(coldist[colact])] for (coldist, colact) in zip(eachcol(pdist), a)]...)

    entropyloss = - Statistics.mean(ent_lpdf[1, :])
    policyloss = - Statistics.mean(d .* Ψ .* ent_lpdf[2,:]) 

    return policyloss + β * entropyloss 
end

struct NaNParamException <: Base.Exception
    model::AbstractModel
    prevmodel::AbstractModel
    #Ψ::Vector{Float32}
    states
    actions
    #discounts
    #values
    #returns
    #β
end

Base.showerror(io::IO, ::NaNParamException) = print(io, "NaNParamException: found NaN values in network parameters")

#Flux.withgradient(e::NaNParamException) = Flux.withgradient((args...) -> policyloss(args...; β = e.β), e.prevmodel.model, e.states, e.actions, e.discounts, e.Ψ)
(e::NaNParamException)() = e.prevmodel(e.states)

struct WorkerException <: Base.Exception
    workerid::Int
    sharedpolicymodel::AbstractModel
    sharedvaluemodel::AbstractModel
    localpolicymodel::AbstractModel
    localvaluemodel::AbstractModel

    innerexception::Base.Exception
end

Base.showerror(io::IO, e::WorkerException) = print(io, "WorkerException: exception raised in worker $(e.workerid): $(e.innerexception)")