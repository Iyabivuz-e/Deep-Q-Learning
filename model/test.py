
import torch
from dqn import DQN
from helpers.encode import state_to_dqn_input
from environments.envs import FrozenLakeEnv

def test(episodes, device, render=False):
        frozenlake_env = FrozenLakeEnv()
        env = frozenlake_env.load_frozenlake_env(render)
        
        # Get the dimensions for the NN
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        policy_dqn = DQN(num_states, num_actions).to(device)
        policy_dqn.load_state_dict(torch.load("frozenlake_dqn.pt"))
        policy_dqn.eval()
        
        for episide in range(episodes):
            current_state = env.reset()[0]
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(state_to_dqn_input(current_state, num_states, device)).argmax().item()
                    
                new_state, reward, terminated, truncated, info = env.step(action) 
                print(
                    f"state={current_state}, "
                    # f"Q={q_values.cpu().numpy()}, "
                    f"action={action}"
                    )
                
                current_state=new_state 
                
        env.close()