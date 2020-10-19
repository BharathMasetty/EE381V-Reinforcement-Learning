import sys
import numpy as np


def nonstationary_bandit(filename, no_bandits, init_q_estimate, init_q_star, std_q_star, std_change_q_star, epsilon,
                         alpha, steps, runs):

    A = np.zeros(steps)
    B = np.zeros(steps)
    C = np.zeros(steps)
    D = np.zeros(steps)

    for run in range(runs):

        q_star = np.ones(no_bandits) * init_q_star
        avg_reward_per_step_simple = []
        optimality_ratio_per_step_simple = []
        total_reward_simple = 0.0
        optimal_action_count_simple = 0
        q_estimate_simple = np.ones(no_bandits) * init_q_estimate
        count_action_simple = np.zeros(no_bandits)
        avg_reward_per_step_const = []
        optimality_ratio_per_step_const = []
        total_reward_const = 0.0
        optimal_action_count_const = 0
        q_estimate_const = np.ones(no_bandits) * init_q_estimate

        for step in range(steps):
            sample_uniform = np.random.uniform()
            if sample_uniform <= epsilon:
                action_simple = np.random.randint(0, no_bandits)
                action_const = action_simple

            else:
                action_simple = np.argmax(q_estimate_simple)
                action_const = np.argmax(q_estimate_const)

            optimal_action = np.argmax(q_star)
            if action_simple == optimal_action:
                optimal_action_count_simple += 1
            if action_const == optimal_action:
                optimal_action_count_const += 1

            optimality_ratio_per_step_simple.append(int(action_simple == optimal_action))
            optimality_ratio_per_step_const.append(int(action_const == optimal_action))
            reward_simple = np.random.normal(q_star[action_simple], std_q_star)
            reward_const = np.random.normal(q_star[action_const], std_q_star)
            total_reward_simple += reward_simple
            total_reward_const += reward_const
            avg_reward_per_step_simple.append(reward_simple)
            avg_reward_per_step_const.append(reward_const)
            count_action_simple[action_simple] += 1
            q_estimate_simple[action_simple] += (reward_simple - q_estimate_simple[action_simple])
            q_estimate_const[action_const] += alpha * (reward_const - q_estimate_const[action_const])
            q_star += np.random.randn(no_bandits) * std_change_q_star

        A += np.array(avg_reward_per_step_simple)
        B += np.array(optimality_ratio_per_step_simple)
        C += np.array(avg_reward_per_step_const)
        D += np.array(optimality_ratio_per_step_const)

    E = A / runs
    F = B / runs
    G = C / runs
    H = D / runs
    np.savetxt(filename, (E, F, G, H))


if __name__ == '__main__':
    filename = sys.argv[1]
    bandits = 10
    init_q_estimate = 0.
    init_q_star = 0.
    std_q_star = 1.
    std_change_q_star = 0.01
    epsilon = 0.1
    alpha = 0.1
    steps = 10000
    runs = 300
    nonstationary_bandit(filename, bandits, init_q_estimate, init_q_star, std_q_star, std_change_q_star, epsilon,
                         alpha, steps, runs)
