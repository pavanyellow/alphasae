import torch
from model import SAEConfig, SparseAutoencoder, DEVICE
from trainer import train_autoencoder



layer_num = 2
resid_stream_dimension = 256
game = 'othello'



PATH = f'activations/{game}_{resid_stream_dimension}_l{layer_num}.pt'
data = torch.load(PATH, map_location=DEVICE)
print(f"Training examples count: {data.shape[0]}")
input_dim = data.shape[1]



NUM_EPOCHS = 12000
batch_size = 8192*2
learning_rate = 1e-4
sae_expansion_factor = 8
l1_penalty = 4



config = SAEConfig(
    input_dim,
    input_dim * sae_expansion_factor,
    layer_num= layer_num,
    l1_penalty=l1_penalty, 
    num_epochs=NUM_EPOCHS, 
    batch_size = batch_size, 
    learning_rate=learning_rate
    )

model = SparseAutoencoder(config)
print(f"No of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


train_autoencoder(model, data, config)

model.eval()

model.save()

