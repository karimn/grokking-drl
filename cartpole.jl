

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

function (e::CartPoleEnv)(action) 
    e.currstep += 1
    e.innerenv(action)
end

istruncated(e::CartPoleEnv) = e.currstep > e.max_steps
nactions(::CartPoleEnv) = 2
spacedim(::CartPoleEnv) = 4 