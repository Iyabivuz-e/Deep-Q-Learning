import torch

def state_to_dqn_input(current_state, num_states, device):
    input_tensor = torch.zeros(num_states, device=device)
    input_tensor[int(current_state)] = 1 # We put 1 to the current state, and the rest remain 0
       
    return input_tensor