from environments.envs import FrozenLakeEnv
import time
from dqn import DQN
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available else "cpu"

print(f"Device: {device}")


class Agent:
    def run(self, is_training=True, render=True): 
        # Create the environment
        frozenlake_env = FrozenLakeEnv()
        env = frozenlake_env.load_frozenlake_env(render)
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        # print(f"States are: {num_states} || Actions are: {num_actions}")
        policy_dqn = DQN(num_states, num_actions).to(device)
        # states = torch.randn(1, state_dim)
        # output = dq_network.forward(states)
        # print(f"Output here:: {output}")
        
        state, info = env.reset()
        print(f"State: {state}")
        print(f"info: {info}")
        
        env.render()

        for episode in range(1000):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            # print(f"Episode: {episode} => State: {state} || Rewar: {reward} || Info: {info}")
            if terminated or truncated:
                state, info = env.reset()
        
        env.close()

if __name__ == "__main__":
    agent = Agent()
    agent.run()
