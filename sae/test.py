import torch
from model import SparseAutoencoder
from train import train_autoencoder, SAETrainConfig


input_dim = 4
hidden_dim = 24

real_features = 32
sparsity = 0.2
no_active = int(real_features * sparsity)

data = []

iterations = 2**18

directions = torch.randn(real_features, input_dim)
# Generate data
for i in range(iterations):
    indices = torch.randperm(real_features)[:no_active]
    acts = (torch.randint(-2, 2, (no_active,1)) * directions[indices]).sum(dim=0)
    data.append(acts)

data = torch.stack(data)

model = SparseAutoencoder(input_dim, hidden_dim)
train_autoencoder(model, data, SAETrainConfig(input_dim, hidden_dim, num_epochs=2**15, batch_size=2**5, l1_penalty=2, learning_rate=1e-4))

