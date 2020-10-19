import numpy as np
from policy import Policy


class ValueFunctionWithApproximation(object):
    def __call__(self, s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self, alpha, G, s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()


def semi_gradient_n_step_td(
        env,  # open-ai environment
        gamma: float,
        pi: Policy,
        n: int,
        alpha: float,
        V: ValueFunctionWithApproximation,
        num_episode: int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: # episodes to iterate
    output:
        None
    """
    # TODO: implement this function

    for i in range(num_episode):
        observation = env.reset()
        T = 200
        t = 0
        tau = 0
        # initialize S and R lists for the episode
        S = []
        R = []
        S.append(observation)
        R.append([0])

        while tau != T - 1:
            if t < T:
                # env.render()
                action = pi.action(observation)
                observation, reward, done, info = env.step(action)
                # we have information about St+1, Rt+1, and termination
                S.append(observation)
                R.append(reward)

                if done:
                    T = t + 1

            tau = t - n + 1

            if tau >= 0:
                G = 0

                for j in range(tau + 1, min(tau + n, T) + 1):
                    G = G + (gamma ** (j - tau - 1)) * R[j]

                if tau + n < T:
                    G = G + V(S[tau + n]) * gamma ** n

                V.update(alpha, G, S[tau])

            t = t + 1
