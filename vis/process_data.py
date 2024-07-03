import torch
import json
import random
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from sae.model import SparseAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_activations_to_json(features: torch.Tensor, output_file: str, sae : SparseAutoencoder):

    activation_count, feature_count = features.shape


    data = {
        "total_activations": activation_count,
        "total_features": features.shape[1],
        "features": []
    }

    for i in tqdm(range(feature_count)):
        all_activations = features[:, i]
        positive_indices = torch.where(all_activations > 0.1)[0]
        if positive_indices.shape[0] == 0: 
            continue

        sorted_positive_indices = positive_indices[torch.argsort(all_activations[positive_indices])]
        
        num_samples = 6
        top = sorted_positive_indices[-num_samples:].tolist()[::-1]
                
        if len(positive_indices) >= num_samples:
            random_5 = random.sample(positive_indices.tolist(), num_samples)
        else:
            random_5 = positive_indices.tolist()
        

        move_predictions = torch.matmul(sae.decoder.weight[:,i], policy_head.T)
        top_predictions = torch.argsort(move_predictions, descending=True)[:5]
        bottom_predictions = torch.argsort(move_predictions, descending=False)[:5]
        
                
        feature_data = {
            "id": i,
            "top_5": [{"i": int(idx), "a": f"{all_activations[idx].item():.3f}"} for idx in top],
            "random_5": [{"i": int(idx), "a": f"{all_activations[idx].item():.3f}"} for idx in random_5],
            "activation_frequency": float(positive_indices.shape[0]) / activation_count, 
            "top_moves" : top_predictions.tolist(),
            "bottom_moves": bottom_predictions.tolist()
        }
        data["features"].append(feature_data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"Data preprocessed and saved to {output_file}")


def get_feature_activations(sae: SparseAutoencoder, neuron_activations: torch.Tensor):
    
    feature_activations = torch.zeros(neuron_activations.shape[0], sae.hidden_dim, device=DEVICE)

    batch_size = 2048
    num_batches = neuron_activations.shape[0] // batch_size
    for i in tqdm(range(num_batches), desc="Getting feature activations"):
        batch = neuron_activations[i*batch_size:(i+1)*batch_size]
        _, encoded = sae(batch)
        normalized_encoded = encoded*torch.norm(sae.decoder.weight, p=2, dim=0)
        feature_activations[i*batch_size:(i+1)*batch_size] = normalized_encoded
    
    return feature_activations

def save_boards_to_json(boards: torch.Tensor, output_file: str):
    from othello.OthelloGame import OthelloGame
    game = OthelloGame(6)

    data = {
        "total_boards": len(boards),
        "boards": [],
    }

    for b in tqdm(boards, desc="Saving boards"):
        b = b[0]
        board_data = {"board": b.flatten().tolist()}
        valid_moves = game.getValidMoves(b, 1) 
        board_data["valid_moves"] = np.where(valid_moves == 1)[0].tolist()
        data["boards"].append(board_data)

    with open(output_file, 'w') as f:
        json.dump(data, f)

    print(f"Boards saved to {output_file}")

def process_data(layer_num, l1_penalty, sae_expansion_factor, input_dim):
    print(f"Preprocessing data for layer {layer_num}, l1_penalty {l1_penalty}, and sae_expansion_factor {sae_expansion_factor}")
    sae = SparseAutoencoder.load("../sae", layer_num, input_dim, input_dim*sae_expansion_factor, l1_penalty)
    neuron_activations = torch.load(f"activations/othello/l{layer_num}_neurons.pt", map_location = DEVICE)[:BOARDS_COUNT_TO_USE]
    feature_activations = get_feature_activations(sae, neuron_activations)
    

    save_activations_to_json(feature_activations, f"data/layer_{layer_num}_penalty_{l1_penalty}_count_{input_dim*sae_expansion_factor}.json", sae)

    

layers = [0,1,2,3]
l1_penaties = [2,5,10]
sae_expansion_factors = [2,4,8,16]
INPUT_DIM = 256
BOARDS_COUNT_TO_USE = 300*1000
policy_head = torch.load("activations/othello/policy_head.pt",map_location = DEVICE).weight.data

boards = torch.load(f"activations/othello/boards.pt")[:BOARDS_COUNT_TO_USE]
save_boards_to_json(boards, "data/boards.json")

# for layer_num in [2]:
#     for l1_penalty in [1,2,3,4,5,10]:
#         for sae_expansion_factor in [2,4,8,16]:
#             process_data(layer_num, l1_penalty, sae_expansion_factor, input_dim = INPUT_DIM)


process_data(layer_num= 2, l1_penalty= 5, sae_expansion_factor= 16, input_dim= INPUT_DIM)
            
            
















