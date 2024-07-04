import sys
import argparse
sys.path.append('..')

from alphazero.Coach import SelfPlayConfig
from othello.OthelloGame import OthelloGame as Game
from othello.OthelloPlayers import RandomPlayer, GreedyPlayer, HumanPlayer
from othello.NetworkWrapper import NNetWrapper
from alphazero.utils import NNetWrapperConfig
from alphazero.Arena import Arena, Player

folder = 'pretrained/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'

game = Game(6)

def get_player(folder, filename, temp=1):
    network = NNetWrapper(game, NNetWrapperConfig())
    network.load_checkpoint(folder, filename)
    return Player(game, SelfPlayConfig(numMCTSSims=32), network, temp)

random_player = RandomPlayer(game)
greedy_player = GreedyPlayer(game)
human_player = HumanPlayer(game)
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Play Othello against AI or watch AI vs AI')
parser.add_argument('--human', action='store_true', help='Play as human against the best AI')
parser.add_argument('--games', type=int, default=10, help='Number of games to play')
args = parser.parse_args()

if args.human:
    opponent = human_player
    num_games = 1  # When playing as human, we typically want to play just one game
else:
    opponent = second_best_player
    num_games = args.games

arena = Arena(best_player, opponent, game, display=Game.display)

print(arena.playGames(num_games, verbose=False, print_final_board=True))