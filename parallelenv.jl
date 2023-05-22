struct ParallelEnv{E <: AbstractEnv} <: AbstractAsyncEnv 
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

ParallelEnv(e::ParallelEnv) = ParallelEnv(e.env, e.nworkers) 

innerenv(e::ParallelEnv) = e.env

function Base.put!(c::Channel, e::E) where E <: AbstractEnv 
    status = (state(e), reward(e), is_terminated(e), istruncated(e))

    put!(c, status)
end

function envworker(wid::Int, localenv::E; input::Channel, output::Channel) where E <: AbstractEnv
    exit = false

    while !exit
        cmd = take!(input)

        if cmd == :reset
            reset!(localenv)
        elseif cmd == :step
            a = take!(input)
            localenv(a)
        elseif cmd == :query
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
RLBase.is_terminated(e::ParallelEnv) = any(e.terminated)
istruncated(e::ParallelEnv) = any(e.truncated)

is_terminateds(e::ParallelEnv) = e.terminated
istruncateds(e::ParallelEnv) = e.truncated

function RLBase.reset!(e::ParallelEnv, i)
    put!.(e.cmdchannels[i], :reset)
    refreshinfo!(e, i)
end

RLBase.reset!(e::ParallelEnv) = reset!(e, :)  

function (env::ParallelEnv)(actions) 
    put!.(env.cmdchannels, :step)
    put!.(env.cmdchannels, actions)
    refreshinfo!(env)
end

nactions(e::ParallelEnv) = nactions(e.env) 
spacedim(e::ParallelEnv) = spacedim(e.env) 

function refreshinfo!(e::ParallelEnv, i)
    put!.(e.cmdchannels[i], :query)

    for (wid, s) in zip(axes(e.inchannels)[1][i], take!.(e.inchannels[i]))
        e.states[wid], e.rewards[wid], e.terminated[wid], e.truncated[wid] = s 
    end
end

refreshinfo!(e::ParallelEnv) = refreshinfo!(e, :)

close!(e::ParallelEnv) = put!.(e.cmdchannels, :exit)
