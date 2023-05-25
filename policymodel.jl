function selectaction(m::AbstractPolicyModel, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return rand.(rng, Distributions.Categorical.(copy(colp) for colp in eachcol(p)))
end

function selectgreedyaction(m::AbstractPolicyModel, state; usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return vec(getindex.(argmax(p, dims = 1), 1))
end

function evaluate(m::M, env::AbstractEnv; nepisodes = 1, greedy = true, rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) where M <: AbstractPolicyModel
    rs = []

    env = deepcopy(env)

    for _ in 1:nepisodes
        reset!(env)
        s, d = state(env), false
        push!(rs, 0)

        while !d 
            a = greedy ? selectgreedyaction(m, s; usegpu) : selectaction(m, s; rng, usegpu)
            env(only(a))
            s, r, d = state(env), reward(env), is_terminated(env)
            rs[end] += r
        end
    end

    return mean(rs), std(rs)
end


