import sys
sys.path.append('..')
from alphazero.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from othello.OthelloGame import OthelloGame
from alphazero.utils import NNetWrapperConfig, Activations


acts = Activations(num_layers= 4)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.fc(x)))
        return out + residual

class OthelloModel(nn.Module):
    def __init__(self, game: OthelloGame, num_layers=4, hidden_dim=256):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        super(OthelloModel, self).__init__()

        self.fc1 = nn.Linear(self.board_x * self.board_y, hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(hidden_dim)

        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor, config: NNetWrapperConfig):
        global acts
        
        new_board_state= False
        if config.collect_resid_activations or config.collect_sae_feature_activations:
            board_string = ' '.join(map(lambda x: str(x.item()), s[0].flatten()))
            if board_string not in acts.current_boards:
                acts.current_boards.add(board_string)
                acts.boards.append(s.detach())
                new_board_state = True


        s = s.view(-1, self.board_x * self.board_y)
        s = F.relu(self.fc_bn1(self.fc1(s)))

        for layer_num, res_block in enumerate(self.res_blocks):
            s = res_block(s)
            if not new_board_state:
                # Duplicate state
                continue
            
            # Collect residual stream activations for sae training
            if config.collect_resid_activations:
                # Note batch size is one here during inference
                acts.neurons[layer_num].append(s[0].detach().cpu().numpy())

            # Get SAE activations and boards for feature visualization
            if config.sae is not None and layer_num == config.layer_num:
                decoded, encoded = config.sae(s)
                normalised_activations = encoded*torch.norm(config.sae.decoder.weight, p=2, dim=0)

                if config.collect_sae_feature_activations: 
                    acts.features[layer_num].append(normalised_activations[0].detach().cpu().numpy())

                if config.replace_sae_activations:
                    s = decoded
                    #s = torch.zeros(s.shape) # check for baseline


            

        policy = self.policy_head(s)
        value = self.value_head(s)

        return F.log_softmax(policy, dim=1), torch.tanh(value)
