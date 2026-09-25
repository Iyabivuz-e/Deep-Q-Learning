import torch
from .encode import state_to_dqn_input

def optimize_policy_network(mini_batch, policy_dqn, target_dqn, device, loss_fn, optimizer, discounted_factor):
        num_states = policy_dqn.fc1.in_features
        
        current_q_list = [] # The output of the policy dqn (the states)
        target_q_list  = [] # The output of the target dqn (the states)
        
        for current_state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                target = reward + (discounted_factor * target_dqn(state_to_dqn_input(new_state, num_states, device)).max())
                target = target.to(device)
            
            # Get the current/target q lists(the output state)
            current_q = policy_dqn(state_to_dqn_input(current_state, num_states, device))
            current_q_list.append(current_q[action])
            
            target_q = target_dqn(state_to_dqn_input(current_state, num_states, device))
            target_q[action] = target
            target_q_list.append(target_q[action])
        
        # Compute loss 
        loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        
        # We compute backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()