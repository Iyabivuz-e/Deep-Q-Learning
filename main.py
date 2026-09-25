import yaml
import torch
import torch.nn as nn
from model.test import test
from environments.envs import FrozenLakeEnv
from model.train import train


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
        self.episodes           = hyperparams["episodes"]
                
        self.render = render
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        # self.ACTIONS = ['L', 'D', 'R', 'U']   # This helps to see where the robot is going    
        
        self.frozenlake_env = FrozenLakeEnv()
        
    ## we train the model
    def run(self):
        train(
            env=self.frozenlake_env.load_frozenlake_env(self.render),
            device=device, learning_rate=self.learning_rate,
            optimizer=self.optimizer, epsilon_init=self.epsilon_init,
            replay_buffer_size=self.replay_buffer_size, episodes=self.episodes,
            loss_fn=self.loss_fn, mini_batch_size=self.mini_batch_size,
            discounted_factor=self.discounted_factor, epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min,
            network_sync_rate=self.network_sync_rate, is_training=True
            
        )


if __name__ == "__main__":
    agent = Agent("frozenlake", render=True)
    agent.run()
    test(episodes=4, device=device, render=True)
