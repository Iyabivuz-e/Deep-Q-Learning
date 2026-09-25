import yaml
import torch
import torch.nn as nn

class Agent:
    def __init__(self,  hyperparams_set, render=True):
        super(Agent, self).__init__()
        
        ## Hyperparams settings...
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
        
        # self.ACTIONS = ['L', 'D', 'R', 'U']   # This helps to see where the robot is going    

        