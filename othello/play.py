import sys
sys.path.append('..')

from othello.OthelloGame import OthelloGame as Game
from othello.OthelloPlayers import  RandomPlayer, GreedyPlayer, HumanPlayer
from othello.pytorch.NNet import NNetWrapper as NNet
from alphazero.utils import NNetWrapperConfig, dotdict, print_size_of
from alphazero.Arena import Arena, Player


folder = 'temp/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'

use_pretrained = True
if use_pretrained:
    folder = 'pretrained/'

game = Game(6)

def get_player(folder, filename , temp = 1):
    nnet = NNet(game, NNetWrapperConfig())
    nnet.load_checkpoint(folder, filename)
    args = dotdict({'numMCTSSims': 2, 'cpuct': 1,})
    return Player(game, args, nnet, temp)

# all players
random_player = RandomPlayer(game)
greedy_player = GreedyPlayer(game)
human_player = HumanPlayer(game)
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)

arena = Arena(best_player, random_player, game, display=Game.display)


print(arena.playGames(10, verbose=True, print_final_board=True))


