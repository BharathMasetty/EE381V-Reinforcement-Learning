import numpy as np
from algo import ValueFunctionWithApproximation
import math
import gym


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


class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method **Create the tiling and coding here**
        super().__init__()
        self.num_tilings = num_tilings
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width

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

        tilings = create_tilings(state_ranges, num_tilings, bins, offsets, self.tile_width)
        self.tilings = tilings

        # for each tile in a tiling, a weight has to be initialize
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in self.tilings]
        self.weights = [np.ones(shape=state_size) for state_size in self.state_sizes]

    def __call__(self, s):
        # TODO: implement this method
        val = 0

        state_coding = get_tile_coding(s, self.tilings)

        for i in range(self.num_tilings):
            val = val + self.weights[i][tuple(state_coding[i])]

        return val

    def update(self, alpha, G, s_tau):
        # TODO: implement this method

        state_coding = get_tile_coding(s_tau, self.tilings)
        temp = self.__call__(s_tau)
        delta = alpha * (G - temp)

        for i in range(self.num_tilings):
            self.weights[i][tuple(state_coding[i])] += delta

        return None


env = gym.make("MountainCar-v0")
#
V = ValueFunctionWithTile(
    env.observation_space.low,
    env.observation_space.high,
    num_tilings=10,
    tile_width=np.array([.45, .035]))

# print(V.tilings)
#
# S = [-.5, 0]

# print(V.tilings)
# S = [-.5, 0]
# state_coding = get_tile_coding(S, V.tilings)
# print(state_coding)

