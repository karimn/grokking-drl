function TD_lambda(env, V, policy, alpha, lambda, num_episodes)
    for i in 1:num_episodes
        S = env.reset()  # initialize state
        eligibility = Dict()  # initialize eligibility trace
        
        while true
            A = policy(S)  # select action based on policy
            S_next, R, done, info = env.step(A)  # take action and observe next state and reward
            
            delta = R + env.gamma * V[S_next] - V[S]  # calculate TD error
            
            if haskey(eligibility, S)
                eligibility[S] += 1
            else
                eligibility[S] = 1
            end
            
            for s in keys(V)
                V[s] += alpha * delta * eligibility[s]  # update value function for all visited states
                eligibility[s] *= env.gamma * lambda  # decay eligibility trace for all states
            end
            
            if done
                break
            end
            
            S = S_next  # update state
        end
    end
    
    return V
end

function SARSA_λ(env, Q, ε, α, λ, num_episodes)
    for i in 1:num_episodes
        S = env.reset()  # initialize state
        A = ε_greedy_action(Q, S, ε)  # select action based on ε-greedy policy
        eligibility = Dict()  # initialize eligibility trace
        
        while true
            S_next, R, done, info = env.step(A)  # take action and observe next state and reward
            A_next = ε_greedy_action(Q, S_next, ε)  # select next action based on ε-greedy policy
            
            δ = R + env.γ * Q[S_next, A_next] - Q[S, A]  # calculate TD error
            
            if haskey(eligibility, (S, A))
                eligibility[(S, A)] += 1
            else
                eligibility[(S, A)] = 1
            end
            
            for (s, a) in keys(Q)
                Q[s, a] += α * δ * eligibility[(s, a)]  # update Q-value for all visited state-action pairs
                eligibility[(s, a)] *= env.γ * λ  # decay eligibility trace for all state-action pairs
            end
            
            if done
                break
            end
            
            S, A = S_next, A_next  # update state and action
        end
    end
    
    return Q
end

function Dyna_Q(env, Q, ε, α, γ, n, num_episodes)
    model = Dict()  # initialize the model
    
    for i in 1:num_episodes
        S = env.reset()  # initialize state
        
        while true
            A = ε_greedy_action(Q, S, ε)  # select action based on ε-greedy policy
            S_next, R, done, info = env.step(A)  # take action and observe next state and reward
            
            # update Q-value
            Q[S, A] += α * (R + γ * maximum([Q[S_next, a] for a in keys(env.action_space)]) - Q[S, A])
            
            # update model
            if !haskey(model, S)
                model[S] = Dict()
            end
            model[S][A] = (R, S_next)
            
            # plan
            for j in 1:n
                S_sample = sample(keys(model))  # sample state from model
                A_sample = sample(keys(model[S_sample]))  # sample action from model
                R_sample, S_next_sample = model[S_sample][A_sample]  # get reward and next state from model
                
                # update Q-value
                Q[S_sample, A_sample] += α * (R_sample + γ * maximum([Q[S_next_sample, a] for a in keys(env.action_space)]) - Q[S_sample, A_sample])
            end
            
            if done
                break
            end
            
            S = S_next  # update state
        end
    end
    
    return Q
end

