from collections import defaultdict
import torch
from model import SAEConfig, SparseAutoencoder, DEVICE
from train import train_autoencoder
import json



layer_num = 2
resid_stream_dimension = 256
game = 'othello'



PATH = f'activations/{game}_{resid_stream_dimension}_l{layer_num}.pt'
data = torch.load(PATH, map_location=DEVICE)
#data = data
print(f"Training examples count: {data.shape[0]}")
input_dim = data.shape[1]



NUM_EPOCHS = 12000
batch_size = 8192*2
learning_rate = 1e-4
#sae_expansion_factor = 32



# config = SAEConfig(
#     input_dim,
#     input_dim * sae_expansion_factor,
#     layer_num= layer_num,
#     l1_penalty=5, 
#     num_epochs=NUM_EPOCHS, 
#     batch_size = batch_size, 
#     learning_rate=learning_rate
#     )

# model = SparseAutoencoder(config)
# print(f"No of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


# train_autoencoder(model, data, config)

# model.eval()

# model.save()



# for l1_penalty in [3]:
#     for sae_expansion_factor in [32]:
#         config = SAEConfig(
#             input_dim,
#             input_dim*sae_expansion_factor,
#             layer_num= layer_num,
#             l1_penalty=l1_penalty, 
#             num_epochs=NUM_EPOCHS, 
#             batch_size = batch_size, 
#             learning_rate=learning_rate
#             )

#         model = SparseAutoencoder(config)
#         print(f"No of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


#         train_autoencoder(model, data, config)

#         model.eval()

#         model.save()