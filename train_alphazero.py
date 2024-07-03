from alphazero.Coach import Coach, SelfPlayConfig
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper
from othello.OthelloPlayers import  RandomPlayer, GreedyPlayer
from alphazero.utils import *


config = SelfPlayConfig(numIters=15, numEps=10, numMCTSSims=32,  checkpoint='othello/temp/')


game = Game(6)
network = NNetWrapper(game, config = NNetWrapperConfig())

alphazero = Coach(game, network, config, random_player=RandomPlayer(game), greedy_player=GreedyPlayer(game))
alphazero.learn()