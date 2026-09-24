import time
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from dqn import DQN
from environments.envs import FrozenLakeEnv
from experience_replay_buffer import ReplayMemory


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available else "cpu"

print(f"Device: {device}")


class Agent:
    def __init__(self, hyperparams_set, render=True):
        with open("hyperparams.yaml", "r") as file:
            all_hyperparams_set = yaml.safe_load(file)
            hyperparams = all_hyperparams_set[hyperparams_set] # we get the hyperparams we need(atari, frozenlake, ...)
        
        self.replay_buffer_size = hyperparams["replay_buffer_size"]
        self.mini_batch_size    = hyperparams["mini_batch_size"]
        self.epsilon_init       = hyperparams["epsilon_init"]
        self.epsilon_decay      = hyperparams["epsilon_decay"] # Epsilon decay rate
        self.epsilon_min        = hyperparams["epsilon_min"]
        self.network_sync_rate  = hyperparams["network_sync_rate"] # Number of steps before syncing policy and target network
        self.learning_rate      = hyperparams["learning_rate"]
        self.discounted_factor  = hyperparams["discounted_factor"] # For calculating the target
        
        self.render = render
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        self.ACTIONS = ['L', 'D', 'R', 'U']   # This helps to see where the robot is going    
            
    ## We train the network
    def run(self, is_training=True): 
        # Create the environment
        frozenlake_env = FrozenLakeEnv()
        env = frozenlake_env.load_frozenlake_env(self.render)
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        target_dqn = DQN(num_states, num_actions).to(device)
        
        # We copy the policy net params to the target net
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        ##### For debugging: Print the dqn results
        # print("Policy (Random, before training): ")
        # self.print_dqn(policy_dqn)
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
        
        ## Keep track
        rewards_per_episode = []
        epsilon_history     = []
        
        # Count steps so that we can sync
        step_count = 0
        
        # If we are training, we initialize the memory. otherwise, no use
        if is_training:
            memory = ReplayMemory(self.replay_buffer_size)
            epsilon = self.epsilon_init # We initialize the absilon.
        # states = torch.randn(1, state_dim)
        # output = dq_network.forward(states)
        # print(f"Output here:: {output}")
        
        # print(f"State: {current_state}")
        # print(f"info: {info}")
        
        # env.render()
        
        for episode in range(200):
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
                        # print(f"==== Action before exploit: {action.shape}")
                        # action = policy_dqn(current_state.unsqueeze(dim=0)).squeeze().argmax() # Exploitation
                        action = policy_dqn(state_to_dqn_input(current_state, num_states)).argmax()
                        
                        ## We unsqueeze it to put it in a 2d dimention then squeeze it to grad the max value
                        
                # print(f"state: {type(current_state)}")
                # print(f"Action: {type(action)}")
                
                new_state, reward, terminated, truncated, info = env.step(action.item()) # removed item()
                print(f"new state: {new_state} || reward: {reward} ")
                # print(f"new state type: {type(new_state)} || reward type: {type(reward)} ")
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                
                if is_training:
                    memory.append_to_buffer((current_state, action, new_state, reward, terminated))
                
                accumulated_reward +=reward # We correct the current reward
                current_state = new_state # We update the current step
                step_count +=1 ## We increment the step counter
                
                if terminated or truncated:
                    current_state, info = env.reset()
                    break
            # if we have a reward of this episode, we put it in the list.
            if accumulated_reward == 1:
                print(f"Episode: {episode} => New state: {current_state} || Reward: {reward} || Info: {info}")
                rewards_per_episode.append(accumulated_reward)
                
            ## We check whether we have enough training experiences from the buffer and then we sample.
            # We also check if we have gotten atleast a reward, otherwise we would be sampling from useless.
            if len(memory) > self.mini_batch_size and sum(rewards_per_episode) > 0:
                mini_batch = memory.sample_from_buffer(self.mini_batch_size)
                self.optimize_policy_network(mini_batch, policy_dqn, target_dqn)
            
            ## We sync the networks if we have reached the sync rate.
            if step_count > self.network_sync_rate:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # we make sure that we wont go below the min epsilon
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count=0
            print(f"Episode: {episode} DONE")
        # print(f"Rewards per episode: {torch.sum(rewards_per_episode)} || Epsilon History: {epsilon_history}")
        print(f"Rewards per episode: {rewards_per_episode} || Epsilon History: {epsilon_history}")
            
        env.close()
        
        torch.save(policy_dqn.state_dict(), "frozenlake_dqn.pt")
        
        
        
    def optimize_policy_network(self, mini_batch, policy_qn, target_qn):
        num_states = policy_qn.fc1.in_features
        
        current_q_list = [] # The output of the policy dqn (the states)
        target_q_list  = [] # The output of the target dqn (the states)
        
        for current_state, action, new_state, reward, terminated in mini_batch:
            # print(f"Before----Device: {device}")
            # current_state = current_state.to(device)
            # new_state.to(device)
            # reward.to(device)
            
            # print(f"Afgter-----Device: {device}")

            
            # If the agent reached the goal or fell in the hall. then the target is the reward gotten so far
            # Otherwise, we should calculate the target (reward + gamma * argmax(dqn(state)))
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                target = reward + (self.discounted_factor * policy_qn(state_to_dqn_input(current_state, num_states)).max())
                target = target.to(device)
            
            # Get the current/target q lists(the output state)
            current_q = policy_qn(state_to_dqn_input(current_state, num_states))
            current_q_list.append(current_q)
            
            target_q = target_qn(state_to_dqn_input(current_state, num_states))
            target_q[action] = target
            target_q_list.append(target_q)
        
        # Compute loss 
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        
        # We compute backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
           
    def test(self, episodes, is_slippery=False):
        frozenlake_env = FrozenLakeEnv()
        env = frozenlake_env.load_frozenlake_env(self.render)
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        policy_dqn.load_state_dict(torch.load("frozenlake_dqn.pt"))
        policy_dqn.eval()
        
        for episide in range(episodes):
            current_state = env.reset()[0]
            self.terminated = False
            self.truncated = False
            
            while not self.terminated and not self.truncated:
                with torch.no_grad():
                    action = policy_dqn(state_to_dqn_input(current_state, num_states)).argmax().item()
                    
                new_state, reward, terminated, truncated, info = env.step(action) 
                
        env.close()
                
        # target_dqn = DQN(num_states, num_actions).to(device)
        
        
        

def state_to_dqn_input(current_state, num_states):
    input_tensor = torch.zeros(num_states, device=device)
    input_tensor[int(current_state)] = 1 # We put 1 to the current state, and the rest remain 0
       
    return input_tensor


    


if __name__ == "__main__":
    agent = Agent("frozenlake", render=True)
    # agent.run(is_training=True)
    agent.test(4, is_slippery=False)
