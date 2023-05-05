struct EpisodeResult
    totalsteps
    mean100reward
    mean100evalscore
end

function update!(to::Flux.Chain, from::Flux.Chain; tau = 1.0)
    if tau > 0.0
        newparams = [(1 - tau) * toparam + tau * fromparam for (toparam, fromparam) in zip(Flux.params(to), Flux.params(from))]
        Flux.loadparams!(to, newparams)
    end

    return to 
end