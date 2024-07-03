import sys
sys.path.append('..')

from alphazero.Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from othello.OthelloPlayers import  RandomPlayer, GreedyPlayer, HumanPlayer
from alphazero.utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 200,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # Number of moves for which the temperature is 1.0.
    'updateThreshold': 0.55,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 32,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = Game(6)
    nnet = nn(g, config = NNetWrapperConfig())

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    coach = Coach(g, nnet, args, random_player=RandomPlayer(g), greedy_player=GreedyPlayer(g))
    if args.load_model:
        print("Load trainExamples from file")
        coach.loadTrainExamples()
    coach.learn()