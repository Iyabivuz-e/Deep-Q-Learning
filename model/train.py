import torch
import random
from dqn import DQN
from experience_replay_buffer import ReplayMemory
from helpers.encode import state_to_dqn_input
from helpers.optimize import optimize_policy_network

def train(
    env, device,learning_rate, 
    optimizer, epsilon_init, 
    replay_buffer_size,
    episodes,
    loss_fn, 
    mini_batch_size,
    discounted_factor,
    epsilon_decay,
    epsilon_min,
    network_sync_rate,
    is_training=True
    
    ): 
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        target_dqn = DQN(num_states, num_actions).to(device)
        
        # We copy the policy net params to the target net
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)
        
        ## Keep track
        rewards_per_episode = []
        epsilon_history     = []
        
        # Count steps so that we can sync
        step_count = 0
        
        # If we are training, we initialize the memory. otherwise, no use
        if is_training:
            memory = ReplayMemory(replay_buffer_size)
            epsilon = epsilon_init # We initialize the absilon.
        
        for episode in range(episodes):
            print(f" ===== Episode: {episode} DONE ========")
            
            current_state, info = env.reset()
            current_state = torch.tensor(current_state, dtype=torch.float, device=device)
            accumulated_reward = 0.0
            
            while True:
                ## We implement epslon greedy algorithm
                if is_training and random.random() < epsilon: # The random number is chosen between 0 and 1
                    action = env.action_space.sample() # Exploration
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                    
                else:
                    with torch.no_grad(): ## there's an issue with exploiting
                        action = policy_dqn(state_to_dqn_input(current_state, num_states, device=device)).argmax()                        

                new_state, reward, terminated, truncated, info = env.step(action.item()) 
                # print(f"new state: {new_state} || reward: {reward} ")
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                
                if is_training:
                    memory.append_to_buffer((current_state, action, new_state, reward, terminated))
                
                accumulated_reward +=reward # We correct the current reward
                current_state = new_state # We update the current step
                step_count +=1 ## We increment the step counter
                
                if len(memory) > mini_batch_size:
                    mini_batch = memory.sample_from_buffer(mini_batch_size)
                    optimize_policy_network(
                        mini_batch=mini_batch, 
                        policy_dqn=policy_dqn, 
                        target_dqn=target_dqn, 
                        device=device,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        discounted_factor=discounted_factor
                        )
                    epsilon = max(epsilon * epsilon_decay, epsilon_min) # we make sure that we wont go below the min epsilon
                    epsilon_history.append(epsilon)
                    print(f"Epsilon reduction: {epsilon:.4f}")
                    # Sync the Networks
                    if step_count > network_sync_rate:
                        # epsilon = max(epsilon * epsilon_decay, epsilon_min) # we make sure that we wont go below the min epsilon
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        # print(f"*********** Syncing the Networks. ***********")
                        step_count=0
                        
                if terminated or truncated:
                    current_state, info = env.reset()
                    break
            # if we have a reward of this episode, we put it in the list.
            if accumulated_reward == 1:
                print(f"Episode: {episode} => New state: {current_state} || Reward: {reward} || Info: {info.items}")
                rewards_per_episode.append(accumulated_reward)
                
            print(f" ===== Episode: {episode} DONE ========")
        print(f"Rewards per episode: {rewards_per_episode} || Epsilon History: {epsilon_history}")
            
        env.close()
        
        torch.save(policy_dqn.state_dict(), "frozenlake_dqn.pt")