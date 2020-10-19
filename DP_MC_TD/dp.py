

from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy


class OptimisticPolicy(Policy):
    def __init__(self, optActionProb, optPolicy):
        self.optActionProb=optActionProb
        self.optPolicy=optPolicy

    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        return self.optActionProb[state,action]

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        return self.optPolicy[state]


def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    n_s = env.spec.nS
    n_a = env.spec.nA
    trans_mat = env.TD
    reward_matrix = env.R

    delta = theta
    v = initV
    q = np.zeros((n_s, n_a))
    while delta >= theta:
        delta = 0
        for s in range(n_s):
            current_state_val = v[s]
            result = 0
            for a in range(n_a):

                trans = trans_mat[s][a]
                sum_val = 0

                for i in range(len(trans)):

                    next_state = i
                    prob = trans[i]

                    sum_val += (prob*(reward_matrix[s][a][next_state]+(env.spec.gamma * v[next_state])))
                q[s][a] = sum_val

                result += pi.action_prob(s, a)*sum_val

            v[s] = result
            delta = max(delta, abs(v[s]-current_state_val))

    V = v
    Q = q
    return V, Q


def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    n_s = env.spec.nS
    n_a = env.spec.nA
    reward_mat = env.R

    action_prob_opt = np.zeros((n_s, n_a))
    delta = theta
    v = initV
    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    trans_mat = env.TD
    policy = np.zeros(n_s)
    while delta >= theta:
        delta = 0
        for s in range(n_s):
            current_state_val = v[s]
            result=0
            optimal_val=0
            sum_val = np.zeros(n_a)
            for a in range(n_a):

                transition = trans_mat[s][a]

                # print(transition)
                for i in range(len(transition)):
                    next_state = i
                    prob = transition[i]
                    sum_val[a] += (prob*(reward_mat[s][a][next_state]+(env.spec.gamma*v[next_state])))

            max_val = max(sum_val)

            count = np.count_nonzero(sum_val == max_val)
            action_prob_opt[s] = [1/count if v == max_val else 0 for v in sum_val]

            result = np.max(sum_val)
            v[s] = result

            optimal_val_action = np.argmax(sum_val)
            policy[s] = optimal_val_action
            delta = max(delta, abs(v[s]-current_state_val))
    pi = OptimisticPolicy(action_prob_opt, policy)
    V = v
    return V, pi
