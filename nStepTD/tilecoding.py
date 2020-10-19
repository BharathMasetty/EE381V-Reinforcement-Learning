import numpy as np


def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """

    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def create_tilings(features_range, number_tilings, bins, offsets):
    tilings = []
    # for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(features_range)):
            feat_range = features_range[feat_i]
            # tiling for feature 1
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)

        tilings.append(tiling)

    return np.array(tilings)


feature_ranges = [[-1, 1], [2, 5]]  # 2 features
number_tilings = 3
bins = [[10, 10], [10, 10], [10, 10]]  # each tiling has a 10*10 grid
offsets = [[0, 0], [0.2, 1], [0.4, 1.5]]

tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

print(tilings.shape)  # # of tilings X features X bins


# def get_tile_coding(feature, tilings):
#     """
#     feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
#     tilings: tilings with a few layers
#     return: the encoding for the feature on each layer
#     """
#     num_dims = len(feature)
#     feat_codings = []
#     for tiling in tilings:
#         feat_coding = []
#         for i in range(num_dims):
#             feat_i = feature[i]
#             tiling_i = tiling[i]  # tiling on that dimension
#             coding_i = np.digitize(feat_i, tiling_i)
#             feat_coding.append(coding_i)
#         feat_codings.append(feat_coding)
#     return np.array(feat_codings)
#
#
# feature = [0.1, 2.5]
#
# coding = get_tile_coding(feature, tilings)
# print(coding)
