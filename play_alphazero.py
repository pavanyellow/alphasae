from alphazero.Coach import SelfPlayConfig
from othello.OthelloGame import OthelloGame as Game
from othello.OthelloPlayers import  RandomPlayer, GreedyPlayer, HumanPlayer
from othello.pytorch.NNet import NNetWrapper
from alphazero.utils import NNetWrapperConfig, dotdict
from alphazero.Arena import Arena, Player


folder = 'othello/pretrained/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'


game = Game(6)


def get_player(folder, filename , temp = 1):
    network = NNetWrapper(game, NNetWrapperConfig())
    network.load_checkpoint(folder, filename)
    return Player(game, SelfPlayConfig(numMCTSSims=32), network, temp)

random_player = RandomPlayer(game)
greedy_player = GreedyPlayer(game)
human_player = HumanPlayer(game)
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)

arena = Arena(best_player, random_player, game, display=Game.display) # Replace with Human player to play against the AI


print(arena.playGames(10, verbose=True, print_final_board=True))


