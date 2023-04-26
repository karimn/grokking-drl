mutable struct CartPoleEnv <: AbstractEnv
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
