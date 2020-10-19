import gym
import numpy as np
import math
from functools import reduce


# Tiling for single dimension
def create_tiling(feat_range, bins, offset, step_size):
    return np.linspace(feat_range[0], feat_range[1] + step_size, bins + 1)[1:-1] - offset


def create_tilings(feature_ranges, number_tilings, bins, offsets, step_sizes):
    """
    feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings; example: 3tilings
    bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """
    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            step_size = step_sizes[feat_i]
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i], step_size)
            # print(feat_tiling)
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)


def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        # Initialization for tilings
        self.num_tilings = num_tilings
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width
        self.num_actions = num_actions

        dimensions = len(state_low)
        state_ranges = []
        for i in range(dimensions):
            dim_range = [state_low[i], state_high[i]]
            state_ranges.append(dim_range)

        # number of tiles along each dim for a every tiling
        bins = []
        # Contains offsets along each dimension for every tiling
        offsets = []

        for j in range(num_tilings):
            bin = []
            offset = []
            for k in range(dimensions):
                # Number of tiles along kth dimension for a given tiling
                bin.append(math.ceil((state_high[k] - state_low[k]) / tile_width[k]))
                # Offset along kth dimension for a given tiling -- Random offset lesser than tile width
                offset.append(j * tile_width[k] / num_tilings)

            bins.append(bin)
            offsets.append(offset)
            # print(offset)

        # print(bin)
        # print(offsets)
        tilings = create_tilings(state_ranges, num_tilings, bins, offsets, self.tile_width)
        # print(tilings)
        self.tilings = tilings
        self.num_tiles = reduce((lambda x, y: x * y), bins[0])
        self.width = bins[0][0]
        # print(self.width)
        # print(self.num_tiles)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        d = self.num_actions * self.num_tilings * self.num_tiles
        return d

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        active_feature_vector = np.zeros(self.feature_vector_len())
        if done:
            return active_feature_vector
        else:

            state_coding = get_tile_coding(s, self.tilings)
            # print(state_coding)
            for tiling_i in range(self.num_tilings):
                coding = state_coding[tiling_i]

                index = (a * self.num_tilings * self.num_tiles) + (tiling_i * self.num_tiles) + (
                        self.width * (coding[0]) + coding[1])

                active_feature_vector[index] = 1

            return active_feature_vector

        # raise NotImplementedError()


def SarsaLambda(
        env,  # openai gym environment
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size
        X: StateActionFeatureVectorWithTile,
        num_episode: int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    # TODO: implement this function

    for episode in range(num_episode):

        observation = env.reset()
        done = False
        action = epsilon_greedy_policy(observation, done, w)
        x = X(observation, done, action)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0

        while not done:
            # taking action
            observation_dash, reward, done, info = env.step(action)
            action_dash = epsilon_greedy_policy(observation_dash, done, w)
            x_dash = X(observation_dash, done, action_dash)
            Q = np.dot(w, x)
            Q_dash = np.dot(w, x_dash)
            delta = reward + gamma * Q_dash - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_dash
            x = x_dash
            action = action_dash

        # print(w)
    return w


