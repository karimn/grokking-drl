function selectaction(m::AbstractPolicyModel, state; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return rand.(rng, Distributions.Categorical.(copy(colp) for colp in eachcol(p)))
end

function selectgreedyaction(m::AbstractPolicyModel, state; usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return vec(getindex.(argmax(p, dims = 1), 1))
end

function selectaction(m::AbstractPolicyModel, state::Vector; rng::AbstractRNG = Random.GLOBAL_RNG, usegpu = true) 
    p = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu

    return rand(rng, Distributions.Categorical(p))
end

selectgreedyaction(m::AbstractPolicyModel, state::Vector; usegpu = true) = π(m, usegpu ? Flux.gpu(state) : state) |> Flux.cpu |> argmax

function evaluate(m::M, env::AbstractEnv; nepisodes = 1, greedy = true, rng::AbstractRNG = Random.GLOBAL_RNG, copyenv = true, usegpu = true) where M <: AbstractPolicyModel
    rs = []
    steps = []

    if copyenv
        env = deepcopy(env)
    end

    for _ in 1:nepisodes
        reset!(env)
        s, d = state(env), false
        push!(rs, 0)
        push!(steps, 0)

        while !d 
            steps[end] += 1
            a = greedy ? selectgreedyaction(m, s; usegpu) : selectaction(m, s; rng, usegpu)
            #env(only(a)) 
            env(a) 
            s, r, d = state(env), reward(env), is_terminated(env) || istruncated(env)
            rs[end] += r
        end
    end

    return mean(rs), std(rs), mean(steps)
end

function evaluate(m::M, env::AbstractAsyncEnv; nepisodes = 1, greedy = true, rng::AbstractRNG = Random.GLOBAL_RNG, copyenv = true, usegpu = true) where M <: AbstractPolicyModel
    return evaluate(m, innerenv(env); nepisodes, greedy, rng, copyenv, usegpu)

    #=nworkers = env_nworkers(env)
    rs = zeros(nworkers, nepisodes) 
    steps = zeros(Int, nworkers, nepisodes) 

    if copyenv
        env = deepcopy(env)
    end

    for ep in 1:nepisodes
        reset!(env)
        s, d = reduce(hcat, state(env)), is_terminateds(env) .|| istruncateds(env) 

        while !all(d) 
            activeworkers = findall(.!d)
            steps[activeworkers, ep] .+= 1
            a = greedy ? selectgreedyaction(m, s[:, activeworkers]; usegpu) : selectaction(m, s[:, activeworkers]; rng, usegpu)
            env(a, activeworkers) 
            s, r, d = reduce(hcat, state(env)), copy(reward(env)[activeworkers]), is_terminateds(env) .|| istruncateds(env)
            rs[activeworkers, ep] .+= r
        end
    end

    return mean(rs), std(rs), mean(steps)
    =#
end

