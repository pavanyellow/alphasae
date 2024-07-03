import torch

boards = torch.load('boards.pt')
features = torch.load('l0_256.pt')

class FeatureVisualisation:
    def __init__(self, sae_activations, boards, layer_idx):
        self.sae_activations = sae_activations
        self.boards = boards
        self.layer_idx = layer_idx

        self.no_features = self.sae_activations.shape[-1]
        self._build_feature_map()

    def _build_feature_map(self):
        self.feature_map = {}
        for i in range(self.no_features):
            all_activations = self.sae_activations[:, i]

            positive_indices = torch.nonzero(all_activations > 0.2).squeeze()

            sorted_indices = positive_indices[torch.argsort(-all_activations[positive_indices])]
            #print(sorted_indices.shape)

            self.feature_map[i] = [(self.boards[j], all_activations[j]) for j in sorted_indices[:6]]

print(f"Total activations {len(boards)}, featues {features.shape[1]}")

feature_vis = FeatureVisualisation(features, boards, 0)

for board, activation_value in feature_vis.feature_map[90]:
    print(f"Activation Value: {activation_value} for ")
    print("Board:")
    print("")


