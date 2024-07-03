import sys
import os


sys.path.append('..')
from alphazero.Coach import SelfPlayConfig
import alphazero.Arena as Arena
import torch
import time

from othello.OthelloGame import OthelloGame as Game
from othello.OthelloPlayers import  RandomPlayer, GreedyPlayer, HumanPlayer
from othello.NetworkWrapper import NNetWrapper as NNet
from alphazero.utils import NNetWrapperConfig
import numpy as np

from othello.model import acts

config = NNetWrapperConfig(collect_resid_activations= True)

# Script used to collect the residual stream activations that are used to train the sparse autoencoders


folder = 'pretrained/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'

game = Game(6)

def get_player(folder, filename , temp = 1):
    network = NNet(game, NNetWrapperConfig())
    network.load_checkpoint(folder, filename)
    return Arena.Player(game, SelfPlayConfig(numMCTSSims=32), network, temp)

# all players
random_player = RandomPlayer(game)
greedy_player = GreedyPlayer(game)
human_player = HumanPlayer(game)
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)


NUM_GAMES = 500
EPOCHS = 1

for _ in range(EPOCHS):

    start = time.time()

    arena = Arena.Arena(best_player, random_player, game, display=Game.display)
    print(arena.playGames(NUM_GAMES*2, verbose = False, print_final_board= False, use_tqdm= True)) 

    arena = Arena.Arena(best_player, greedy_player, game, display=Game.display)
    print(arena.playGames(NUM_GAMES, verbose = False, print_final_board= False, use_tqdm= True))

    arena = Arena.Arena(best_player, second_best_player, game, display=Game.display)
    print(arena.playGames(NUM_GAMES, verbose = False, print_final_board= False, use_tqdm= True))

    end = time.time()

    print(f"Collected activations from {len(acts.boards)} boards from {NUM_GAMES * 4} games in {end - start} seconds.")

    for i in range(4):
        print(f"collected activations from layer {i} neurons: {len(acts.neurons[i])}")
        print(f"collected activations from layer {i} features: {len(acts.features[i])}")
    
    
    # Used for 1M+ games and loads to disk in batches
    # Used for training sae and doesn't include boards
    for i in range(4):
        PATH = f'../sae/activations/othello_{len(acts.neurons[i][0])}_l{i}.pt'
        if os.path.exists(PATH):
            training_data = torch.load(PATH, map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
        else:
            training_data = torch.tensor([])
        
        current_neurons = torch.from_numpy(np.array(acts.neurons[i]))

        training_data = torch.cat((training_data, current_neurons), dim = 0)

        torch.save(training_data, PATH)

        print(f"Training examples count: {training_data.shape[0]}")

    acts.clear()

# Used to visualise saes and includes boards

# PATH = f'../vis/activations/othello'
# for i in range(4):
#     torch.save(torch.from_numpy(np.array(acts.neurons[i])), f'{PATH}/l{i}_neurons.pt')

# torch.save(acts.boards, f'{PATH}/boards.pt')