import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch Actor-Critic example')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default:543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()
# env.reset(seed=args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

torch.set_default_dtype(torch.float64)


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, num_bs, input_size):
        super(Policy, self).__init__()
        self.num_base_stations = num_bs
        self.affine1 = nn.Linear(input_size, 64)
        self.affine2 = nn.Linear(64, 128)
        # self.affine3 = nn.Linear(128, 256)
        # self.affine4 = nn.Linear(256, 128)

        # actor's layer
        self.action_head = nn.Linear(128, self.num_base_stations)
        # critic's layer
        self.value_head = nn.Linear(128, self.num_base_stations)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x.double().to(self.device)))
        x = F.relu(self.affine2(x.double()))
        # actor: chooses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x.double()), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.value_head(x.double())

        print("State values: ", state_values)
        # Return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


class ACAgent:
    def __init__(self, num_base_stations, device, input_size):
        self.model = Policy(num_bs=num_base_stations, input_size=input_size)
        # Old lr=3e-2
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.model.to(device=device)

    def select_action(self, state):
        # state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)
        # Create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # print("Highest: ", torch.argmax(probs))
        # print("Probabilities: ", m)
        # And sample an action using the distribution
        actions_set = []
        mean_prob = torch.mean(probs)
        for label, p in enumerate(probs):
            # print(f'{label:2}: {100 * p:5.2f}%')
            if mean_prob > p:
                actions_set.append(0)
                print(f'Turning OFF BS ID: {label} --- Raw: {p} Prob: {p*100:.2f} --- Mean: {mean_prob*100:.2f} --- Raw: {mean_prob}')
            else:
                actions_set.append(1)
                print(f'Turning ON BS ID: {label} --- Raw: {p} Prob: {p*100:.2f} --- Mean: {mean_prob*100:.2f} --- Raw: {mean_prob}')

        print("Actions: ", actions_set)
        action = m.sample()
        actions = m.sample_n(132)
        print("Selected action: ", action)
        print("Actions: ", actions)
        print(f"Log probability of action: {action} is : {m.log_prob(action)}")

        # actions_list = []
        # log_probs_list = []
        # for label, p in enumerate(probs):
        #     log_probs_list.append(m.log_prob(torch.tensor(label, dtype=torch.int8)))

        print()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        # The action to take (left or right)
        return action.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values
        # Calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # Calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.mse_loss(value.float().to('cpu'), torch.tensor([R]).float().to('cpu')))
        # Reset gradients
        self.optimizer.zero_grad()
        # Sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        # Perform backprop
        loss.backward()
        # Torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
