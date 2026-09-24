import time
import yaml
import torch
import random
from dqn import DQN
from environments.envs import FrozenLakeEnv
from experience_replay_buffer import ReplayMemory


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available else "cpu"

print(f"Device: {device}")


class Agent:
    def __init__(self, hyperparams_set):
        with open("hyperparams.yaml", "r") as file:
            all_hyperparams_set = yaml.safe_load(file)
            hyperparams = all_hyperparams_set[hyperparams_set] # we get the hyperparams we need(atari, frozenlake, ...)
            print(f"HyperParameters: {hyperparams}")
        
        self.replay_buffer_size = hyperparams["replay_buffer_size"]
        self.mini_batch_size    = hyperparams["mini_batch_size"]
        self.epsilon_init       = hyperparams["epsilon_init"]
        self.epsilon_decay      = hyperparams["epsilon_decay"] # Epsilon decay rate
        self.epsilon_min        = hyperparams["epsilon_min"]
             
    ## We train the network
    def run(self, is_training=True, render=True): 
        # Create the environment
        frozenlake_env = FrozenLakeEnv()
        env = frozenlake_env.load_frozenlake_env(render)
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history     = []
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        
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
        
        for episode in range(1000):
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
                        print(f"==== Action before exploit: {action.shape}")
                        action = policy_dqn.forward(current_state.unsqueeze(dim=0)).squeeze().argmax() # Exploitation
                        ## We unsqueeze it to put it in a 2d dimention then squeeze it to grad the max value
                        print(f"==== Action After exploit: {action.shape}")
                        ##### I have to do one-hot encoding to tell the network of the current state
                new_state, reward, terminated, truncated, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                
                if is_training:
                    memory.append_to_buffer((current_state, action, new_state, reward, terminated))
                
                accumulated_reward +=reward
                current_state = new_state
                if terminated or truncated:
                    current_state, info = env.reset()
                    break
            print(f"Episode: {episode} => New state: {current_state} || Reward: {reward} || Info: {info}")
            rewards_per_episode.append(accumulated_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # we make sure that we wont go below the min epsilon
            print(f"Rewards per episode: {rewards_per_episode} || Epsilon History: {epsilon_history}")

            
        env.close()

if __name__ == "__main__":
    agent = Agent("frozenlake")
    agent.run(is_training=True, render=True)
