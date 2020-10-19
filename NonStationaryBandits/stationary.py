# -*- coding: utf-8 -*-

import numpy as np
import random
from statistics import mean

# Initialize the Final result file
result = np.zeros((4, 300))
alpha = 0.2

for case in range(2):
    # Case 1-> incremental Method
    if case == 1:
        # Initialization of the true action values with mean 0 and std.dev 0.01
        true_Mean = [0.2, -0.8, 1.2, 0.4, 1.1, -1.5, -0.2, -1, 0.8, -0.5]
        true_ActionValue = np.zeros(10)

        # Deciding no of runs and steps for each run
        TotalRuns = 300
        TotalSteps = 10000
        # reward matrix and optimal choice matrix for each run
        best_action = 2  # index of the greedy action
        reward = np.zeros((TotalSteps, TotalRuns))
        optimal_policy = np.zeros((TotalSteps, TotalRuns))

        for x in range(10):
            true_ActionValue[x] = np.random.normal(loc=true_Mean[x], scale=1, size=1)

        # For Each independent run
        for runs in range(TotalRuns):

            # Refreshing expected action value every single Run
            expected_ActionValue = np.zeros(10)
            # Counter for number of times each action is selected in a single run
            n_actions = np.zeros(10)

            # Loop for every step
            for i in range(TotalSteps):

                # choosing the action for the ith step
                m = max(expected_ActionValue)
                # indices of the greedy actions
                greedy_action = [k for k, j in enumerate(expected_ActionValue) if j == m]

                # E-greeedy selection
                rand = random.random()
                if rand <= 0.9:
                    action = random.choice(greedy_action)
                else:
                    action = random.randint(0, 9)

                # Updation of action counter n
                n_actions[action] = n_actions[action] + 1

                # Update optimal_policy matrix
                if action == 2:
                    optimal_policy[i, runs] = 1

                # Update the reward matrix
                r = np.random.normal(loc=true_Mean[action], scale=1, size=1)
                reward[i, runs] = r

                # incremental update of expected action value
                expected_ActionValue[action] = expected_ActionValue[action] + (r - expected_ActionValue[action]) / \
                                               n_actions[action]

        # Now we have a complete reward matrix and optimal policy matrix
        avg_rewards = np.zeros(TotalSteps)
        percent_optimal = np.zeros(TotalSteps)

        for i in range(TotalSteps):
            avg_rewards[i] = mean(reward[i, :])
            percent_optimal[i] = (sum(optimal_policy[i, :]) / TotalRuns) * 100

        # Updating the result file for case 1: Incremental Method
        result[0, :] = avg_rewards[:]
        result[1, :] = percent_optimal[:]

    # Case 2-> Constant alpha
    else:

        # Initialization of the true action values with mean 0 and std.dev 0.01
        true_Mean = [0.2, -0.8, 1.2, 0.4, 1.1, -1.5, -0.2, -1, 0.8, -0.5]
        true_ActionValue = np.zeros(10)

        # Deciding no of runs and steps for each run
        TotalRuns = 300
        TotalSteps = 10000
        # reward matrix and optimal choice matrix for each run
        best_action = 2  # index of the greedy action
        reward = np.zeros((TotalSteps, TotalRuns))
        optimal_policy = np.zeros((TotalSteps, TotalRuns))

        for x in range(10):
            true_ActionValue[x] = np.random.normal(loc=true_Mean[x], scale=1, size=1)

        # For Each independent run
        for runs in range(TotalRuns):

            # Refreshing expected action value every single Run
            expected_ActionValue = np.zeros(10)
            # Counter for number of times each action is selected in a single run
            n_actions = np.zeros(10)

            # Loop for every step
            for i in range(TotalSteps):

                # choosing the action for the ith step
                m = max(expected_ActionValue)
                # indices of the greedy actions
                greedy_action = [k for k, j in enumerate(expected_ActionValue) if j == m]

                # E-greeedy selection
                rand = random.random()
                if rand <= 0.9:
                    action = random.choice(greedy_action)
                else:
                    action = random.randint(0, 9)

                # Updation of action counter n
                n_actions[action] = n_actions[action] + 1

                # Update optimal_policy matrix
                if action == 2:
                    optimal_policy[i, runs] = 1

                # Update the reward matrix
                r = np.random.normal(loc=true_Mean[action], scale=1, size=1)
                reward[i, runs] = r

                # incremental update of expected action value
                expected_ActionValue[action] = expected_ActionValue[action] + alpha * (r - expected_ActionValue[action])

        # Now we have a complete reward matrix and optimal policy matrix
        avg_rewards = np.zeros(TotalSteps)
        percent_optimal = np.zeros(TotalSteps)

        for i in range(TotalSteps):
            avg_rewards[i] = mean(reward[i, :])
            percent_optimal[i] = (sum(optimal_policy[i, :]) / TotalRuns) * 100

        # Updating the result file for case 1: Incremental Method
        result[2, :] = avg_rewards[:]
        result[3, :] = percent_optimal[:]

np.savetxt('result.out', result)
