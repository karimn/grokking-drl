struct ParallelEnv{E <: AbstractEnv} <: AbstractAsyncEnv 
    parallelenvs::Vector{E}
    inchannels::Vector{Channel}
    cmdchannels::Vector{Channel}
    workers::Vector{Task}
    terminated::Vector{Bool}
    truncated::Vector{Bool}
    states::Vector
    rewards::Vector{Union{Nothing, Float32}}
end

env_nworkers(e::ParallelEnv) = length(e.workers)

function ParallelEnv(parallelenvs::Vector{E}) where E <: AbstractEnv
    nworkers = length(parallelenvs)
    inchannels = [Channel(2) for _ in 1:nworkers]
    cmdchannels = [Channel(1) for _ in 1:nworkers]

    envs_and_channels = zip(parallelenvs, inchannels, cmdchannels)
    workers = [@task envworker(wid, localenv, input = outch, output = inch) for (wid, (localenv, inch, outch)) in enumerate(envs_and_channels)]

    for (worker, (_, inch, outch)) in zip(workers, envs_and_channels)
        bind(inch, worker)
        bind(outch, worker)
    end

    schedule.(workers)

    newenv = ParallelEnv{E}(parallelenvs, inchannels, cmdchannels, workers, falses(nworkers), falses(nworkers), Vector(undef, nworkers) , Vector{Float32}(undef, nworkers))

    reset!(newenv)

    return newenv
end

ParallelEnv(e::E, nworkers::Int) where E <: AbstractEnv = ParallelEnv([deepcopy(e) for _ in 1:nworkers])
ParallelEnv(e::ParallelEnv) = ParallelEnv(innerenv(e), env_nworkers(e)) 

function Base.deepcopy_internal(env::ParallelEnv, ::IdDict)
    ParallelEnv(deepcopy.(env.parallelenvs))
end

innerenv(e::ParallelEnv) = e.parallelenvs[1]

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
        # elseif cmd == :pastlimit
        #     put!(output, false)
        else
            exit = true
        end
    end
end

RLBase.action_space(e::ParallelEnv) = action_space(innerenv(e))
RLBase.state(e::ParallelEnv) = e.states
RLBase.state_space(e::ParallelEnv) = state_space(innerenv(e))
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

# BUG if the action are not the right size to broadcast the task will end up waiting and then break on the following action. The below assert should fix this.
function (env::ParallelEnv)(actions) 
    @assert length(actions) == env_nworkers(env)

    put!.(env.cmdchannels, :step)
    put!.(env.cmdchannels, actions)
    refreshinfo!(env)
end

function (env::ParallelEnv)(actions, i::Vector{Int}) 
    @assert all(1 .<= i .<= env_nworkers(env)) 

    put!.(env.cmdchannels[i], :step)
    put!.(env.cmdchannels[i], actions)
    refreshinfo!(env, i)
end

nactions(e::ParallelEnv) = nactions(innerenv(e))
spacedim(e::ParallelEnv) = spacedim(innerenv(e)) 

function refreshinfo!(e::ParallelEnv, i)
    put!.(e.cmdchannels[i], :query)

    for (wid, s) in zip(axes(e.inchannels)[1][i], take!.(e.inchannels[i]))
        e.states[wid], e.rewards[wid], e.terminated[wid], e.truncated[wid] = s 
    end
end

refreshinfo!(e::ParallelEnv) = refreshinfo!(e, :)

close!(e::ParallelEnv) = put!.(e.cmdchannels, :exit)
