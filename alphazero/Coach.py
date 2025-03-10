import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from alphazero.Arena import Arena, Player
from alphazero.MCTS import MCTS
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    numIters: int = 10
    numEps: int = 100
    tempThreshold: int = 15
    updateThreshold: float = 0.55
    maxlenOfQueue: int = 200000
    numMCTSSims: int = 32
    arenaCompare: int = 40
    cpuct: int = 1
    checkpoint: str = './temp/'
    numItersForTrainExamplesHistory: int = 20



class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, random_player = None, greedy_player = None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.nnet.config)  # the competitor network
        self.args : SelfPlayConfig = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.random_player = random_player
        self.greedy_player = greedy_player
        self.p1 = Player(game, args, nnet, temp=1)
        self.p2 = Player(game, args, nnet, temp=1)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                t = tqdm(range(self.args.numEps), desc="Self Play")
                #t = range(self.args.numEps)
                
                for _ in t:
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

                #print(f"Added Num examples: {len(iterationTrainExamples)}")

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            #self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            #print(f"iter {i}, starting training ...")

            self.nnet.train(trainExamples)
            #print(f"iter {i}, starting training ...")

            #print(f"iter {i}, PITTING AGAINST PREVIOUS VERSION")


            arena = Arena(Player(self.game, self.args, self.nnet, temp=0),
                          Player(self.game, self.args, self.pnet, temp=0), self.game)
            new_wins, old_wins, draws = arena.playGames(self.args.arenaCompare)

            print(f"New wins {new_wins}, Previous wins {old_wins}, Draws {draws}")
            if new_wins + old_wins == 0 or float(new_wins) / (old_wins + new_wins) < self.args.updateThreshold:
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print(f'ACCEPTING NEW MODEL with win rate {float(new_wins) *100 / (old_wins + new_wins)} %')
                #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            
            if self.random_player is not None:
                arena = Arena(Player(self.game, self.args, self.nnet, temp=0),
                          self.random_player, self.game)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
                print(f"Agent wins {pwins}, Random wins {nwins}, Draws {draws}")
            if self.greedy_player is not None:
                arena = Arena(Player(self.game, self.args, self.nnet, temp=0),
                          self.greedy_player, self.game)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
                print(f"Agent wins {pwins}, Greedy wins {nwins}, Draws {draws}")
            


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
