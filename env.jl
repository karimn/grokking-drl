

#=mutable struct CartPoleEnv <: AbstractEnv
    pyenv::PyCall.PyObject
    currstate
    lastreward
    terminated::Bool
    truncated::Bool
end

function CartPoleEnv()
    Gym = PyCall.pyimport("gym")
    pyenv = Gym.make("CartPole-v1")
    currstate, _ = pyenv.reset()

    CartPoleEnv(pyenv, currstate, nothing, false, false)
end

action_space(env::CartPoleEnv) = [0, 1] # env.pyenv.action_space
state_space(env::CartPoleEnv) = env.pyenv.observation_space
state(env::CartPoleEnv) = env.currstate
reward(env::CartPoleEnv) = env.lastreward 
is_terminated(env::CartPoleEnv) = env.terminated
is_truncated(env::CartPoleEnv) = env.truncated
reset!(env::CartPoleEnv) = env.pyenv.reset()
nactions(::CartPoleEnv) = 2
spacedim(env::CartPoleEnv) = length(env.currstate)
 
function (env::CartPoleEnv)(action) 
    env.currstate, env.lastreward, env.terminated, env.truncated = env.pyenv.step(action) 
end
=#

mutable struct CartPoleEnv <: RLEnvs.AbstractEnv
    innerenv::RLEnvs.CartPoleEnv
    max_steps::Int
    currstep::Int
end

CartPoleEnv(;max_steps = 500, kwargs...) = CartPoleEnv(RLEnvs.CartPoleEnv(;max_steps, kwargs...), max_steps, 0)

RLEnvs.state(e::CartPoleEnv) = RLEnvs.state(e.innerenv)
RLEnvs.is_terminated(e::CartPoleEnv) = RLEnvs.is_terminated(e.innerenv)
RLEnvs.reward(e::CartPoleEnv) = RLEnvs.reward(e.innerenv)
RLEnvs.state_space(e::CartPoleEnv) = RLEnvs.state_space(e.innerenv)
RLEnvs.action_space(e::CartPoleEnv) = RLEnvs.action_space(e.innerenv)

innerenv(e::CartPoleEnv) = e
is_terminateds(e::CartPoleEnv) = [is_terminated(e)]
istruncateds(e::CartPoleEnv) = [istruncated(e)]
function RLEnvs.reset!(e::CartPoleEnv, id::Array{Int, 1}) 
    @assert id[1] == 1 
    reset!(e)
end

function RLEnvs.reset!(e::CartPoleEnv) 
    e.currstep = 0
    RLEnvs.reset!(e.innerenv)
end

function (e::CartPoleEnv)(action::Int) 
    e.currstep += 1
    act!(e.innerenv, action)
end

(e::CartPoleEnv)(action::Vector{Int}) = e(only(action))

istruncated(e::CartPoleEnv) = e.currstep > e.max_steps
nactions(::CartPoleEnv) = 2
spacedim(::CartPoleEnv) = 4 

mutable struct PendulumEnv <: RLEnvs.AbstractEnv
    innerenv::Union{RLEnvs.PendulumEnv, RLEnvs.ActionTransformedEnv}
    max_steps::Int
    currstep::Int

    PendulumEnv(; max_steps = 200, kwargs...) = new(RLEnvs.PendulumEnv(;max_steps, kwargs...), max_steps, 0)

    function PendulumEnv(nnactbounds; max_steps = 200, kwargs...) 
        innerpendenv = RLEnvs.PendulumEnv(; max_steps, kwargs...)
        actspace = action_space(innerpendenv)

        new(RLEnvs.ActionTransformedEnv(RLEnvs.PendulumEnv(;max_steps, kwargs...); 
                                        action_mapping = a -> ((a - nnactbounds[1]) * (actspace.right - actspace.left) / (nnactbounds[2] - nnactbounds[1])) + actspace.left), 
            max_steps, 0)
    end
end

RLEnvs.state(e::PendulumEnv) = RLEnvs.state(e.innerenv)
RLEnvs.is_terminated(e::PendulumEnv) = RLEnvs.is_terminated(e.innerenv)
RLEnvs.reward(e::PendulumEnv) = RLEnvs.reward(e.innerenv)
RLEnvs.state_space(e::PendulumEnv) = RLEnvs.state_space(e.innerenv)
RLEnvs.action_space(e::PendulumEnv) = RLEnvs.action_space(e.innerenv)

innerenv(e::PendulumEnv) = e

function RLEnvs.reset!(e::PendulumEnv) 
    e.currstep = 0
    RLEnvs.reset!(e.innerenv)
end

(e::PendulumEnv)(action::AbstractArray) = e(only(action))

function (e::PendulumEnv)(action::Float32) 
    e.currstep += 1
    act!(e.innerenv, action)
end

istruncated(e::PendulumEnv) = e.currstep > e.max_steps
nactions(::PendulumEnv) = 1
spacedim(::PendulumEnv) = 3 

mutable struct GymnasiumEnv{T} <: AbstractEnv
    pyenv::PyCall.PyObject
    currstate::Vector{T}
    lastreward::Union{T, Nothing}
    terminated::Bool
    truncated::Bool

    function GymnasiumEnv{T}(envstr::AbstractString; rendermode = nothing) where T
        # Use python-mujoco=2.3.3 to work with gymnasium: pyimport_conda("mujoco", "python-mujoco=2.3.3")
        Gym = PyCall.pyimport("gymnasium") 
        pyenv = Gym.make(envstr, render_mode = rendermode)
        currstate, _ = pyenv.reset()

        new{T}(pyenv, currstate, nothing, false, false)
    end
end

macro gymenv_str(e::AbstractString) 
    GymnasiumEnv{Float32}(e)
end

function RLEnvs.state_space(env::GymnasiumEnv) 
    os = env.pyenv.observation_space

    reduce(×, DomainSets.Interval.(os.low, os.high))
end

function RLEnvs.action_space(env::GymnasiumEnv)
    as = env.pyenv.action_space

    reduce(×, DomainSets.Interval.(as.low, as.high))
end 

RLEnvs.state(env::GymnasiumEnv) = env.currstate
RLEnvs.reward(env::GymnasiumEnv) = env.lastreward 
RLEnvs.is_terminated(env::GymnasiumEnv) = env.terminated || env.truncated
istruncated(env::GymnasiumEnv) = env.truncated
function RLEnvs.reset!(env::GymnasiumEnv) 
    env.pyenv.reset()
    env.terminated, env.truncated = false, false
end

nactions(env::GymnasiumEnv) = action_space(env) |> DomainSets.dimension 
spacedim(env::GymnasiumEnv) = state_space(env) |> DomainSets.dimension 
 
function (env::GymnasiumEnv)(action) 
    if action ∉ action_space(env)
        throw(DomainError(action, "action must fall within an environment's action space"))
    end

    env.currstate, env.lastreward, env.terminated, env.truncated = env.pyenv.step(action) 
end
