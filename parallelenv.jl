struct ParallelEnv{E} <: AbstractEnv where {E <: AbstractEnv}
    env::E
    nworkers::Int
    inchannels::Vector{Channel}
    cmdchannels::Vector{Channel}
    workers::Vector{Task}
    terminated::Vector{Bool}
    truncated::Vector{Bool}
    states::Vector
    rewards::Vector{Float32}
end

function ParallelEnv(e::E, nworkers::Int) where E <: AbstractEnv
    inchannels = [Channel(2) for _ in 1:nworkers]
    cmdchannels = [Channel(1) for _ in 1:nworkers]
    channels = zip(inchannels, cmdchannels)
    workers = [@task envworker(wid, deepcopy(e), input = outch, output = inch) for (wid, (inch, outch)) in enumerate(channels)]

    for (worker, (inch, outch)) in zip(workers, channels)
        bind(inch, worker)
        bind(outch, worker)
    end

    schedule.(workers)

    newenv = ParallelEnv{E}(e, nworkers, inchannels, cmdchannels, workers, falses(nworkers), falses(nworkers), Vector(undef, nworkers) , Vector{Float32}(undef, nworkers))

    reset!(newenv)

    return newenv
end

Base.show(io, e::ParallelEnv) = print(io, "[ParallelEnv with $(e.nworkers) workers]")

Base.put!(c::Channel, e::E) where E <: AbstractEnv = put!(c, (state(e), reward(e), is_terminated(e), istruncated(e)))

function envworker(workerid::Int, localenv::E; input::Channel, output::Channel) where E <: AbstractEnv
    exit = false

    while !exit
        cmd = take!(input)

        if cmd == :reset
            reset!(localenv)
            put!(output, localenv) 
        elseif cmd == :step
            a = take!(input)
            localenv(a)
            put!(output, localenv) 
        elseif cmd == :pastlimit
            put!(output, false)
        else
            exit = true
        end
    end
end

RLBase.action_space(e::ParallelEnv) = action_space(e.env)
RLBase.state(e::ParallelEnv) = e.states
RLBase.state_space(e::ParallelEnv) = state_space(e.env)
RLBase.reward(e::ParallelEnv) = e.rewards
RLBase.is_terminated(e::ParallelEnv) = e.terminated
istruncated(e::ParallelEnv) = e.truncated

function RLBase.reset!(e::ParallelEnv)  
    for c in e.cmdchannels
        put!(c, :reset)
    end

    refreshinfo!(e)
end

function (env::ParallelEnv)(actions) 
    for (c, a) in zip(env.cmdchannels, actions)
        put!(c, :step)
        put!(c, a)
    end

    refreshinfo!(env)
end

function refreshinfo!(e::ParallelEnv)
    for (i, c) in enumerate(e.inchannels)
        e.states[i], e.rewards[i], e.terminated[i], e.truncated[i] = take!(c)
    end
end
