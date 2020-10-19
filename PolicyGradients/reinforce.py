from typing import Iterable
import numpy as np
import gym
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
import collections


class ValueFeedNtw(nn.Module):
    def __init__(self, input_size):
        super(ValueFeedNtw, self).__init__()
        self.input_size = input_size
        self.hidden_size = 32
        self.output_size = 1
        self.hidden_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.hidden_layer1(torch.from_numpy(x))
        out = F.relu(out)
        out = self.hidden_layer2(out)
        out = F.relu(out)
        out = self.fc(out)
        return out


class PolicyFeedNtw(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyFeedNtw, self).__init__()
        self.input_size = input_size
        self.hidden_size = 32
        self.output_size = output_size
        self.hidden_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_layer3 = nn.Linear(self.hidden_size, self.output_size)
        self.Softmax = nn.Softmax(dim=1)

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.hidden_layer3.weight)

    def forward(self, x):
        out = self.hidden_layer1(torch.from_numpy(x))
        out = F.relu(out)
        out = self.hidden_layer2(out)
        out = F.relu(out)
        out = self.hidden_layer3(out)
        out = self.Softmax(out)
        return out


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
        # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
        # a_t = tf.constant([1, 2])
        # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
        # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        self.num_states = state_dims
        self.fwn = PolicyFeedNtw(state_dims, num_actions)
        # self.lossfunc = nn.MSELoss()
        self.optimizer = optim.Adam(self.fwn.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss = 0

    def __call__(self, s) -> int:
        s = s.reshape(1, self.num_states).astype('float32')
        q_s_a = self.fwn.forward(s)
        action_probs = q_s_a.detach().numpy()[0]
        action = np.random.choice(range(len(action_probs)), p=action_probs)
        return action
        # return state_value.detach().numpy()[0]

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this s

        s = s.reshape(1, self.num_states).astype('float32')

        # s_tau = s_tau.reshape(1,self.num_states).astype('float32')
        action_probs = self.fwn(s).view(-1)  # [num_actions, ]
        log_prob_of_a = action_probs[a].log()

        self.loss = -log_prob_of_a * delta * gamma_t
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s) -> float:
        return self.b

    def update(self, s, G):
        pass


class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        self.num_states = state_dims
        self.fwn = ValueFeedNtw(state_dims)
        self.lossfunc = nn.MSELoss()
        self.optimizer = optim.Adam(self.fwn.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss = 0

    def __call__(self, s) -> float:
        # TODO: implement this method

        s = s.reshape(1, self.num_states).astype('float32')
        state_value = self.fwn.forward(s)
        return state_value.detach().numpy()[0][0]

    def update(self, s, G):
        # TODO: implement this method
        s_tau = s.reshape(1, self.num_states).astype('float32')
        estim = self.fwn(s_tau)
        target = torch.tensor(float(G)).view(-1, 1)

        self.loss = (self.lossfunc(estim, target))
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    
    G_0s = []
    
    action_space = np.arange(env.action_space.n)
    
    for i in range(num_episodes):
        
        state = env.reset()
        done = False
        episode = []
        
        while not done:
            action = action_space[pi(state)]
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward, action))
            state = next_state
            
        rewards = [ep[1] for ep in episode]
        states = [ep[0] for ep in episode]
        actions = [ep[2] for ep in episode]
        # print(len(episode))
        # print(len(rewards))
        for t in range(len(episode)):
            # G = sum(gamma ** (k - t-1) * rewards[k] for k in range(t+1, len(episode)))            
            G = 0 
            for k in range(t+1, len(episode)):
            	G = G + gamma**(k-t-1) * rewards[k]

            V.update(states[t], G)
            delta = G - V(states[t])
            pi.update(states[t], actions[t], gamma, delta)
            if t == 0:
                G_0s.append(G)
                        
    print("Best G 0 is ", max(G_0s))    
    return G_0s
