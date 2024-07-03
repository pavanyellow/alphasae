import logging

from tqdm import tqdm
from alphazero.MCTS import MCTS
import numpy as np
log = logging.getLogger(__name__)


class Player():
    def __init__(self, game, args , nnet, temp) -> None:
        self.game = game
        self.args = args
        self.nnet = nnet
        self.temp = temp
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def reset_search_tree(self):
        self.mcts = MCTS(self.game, self.nnet, self.args)
    
    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=self.temp))

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1  = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False, print_final_board=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """


        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            # TODO better type signature to handle this
            if hasattr(player, "mcts"):
                player.reset_search_tree()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1].play((self.game.getCanonicalForm(board, curPlayer)))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if verbose or print_final_board:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False, print_final_board=False, use_tqdm=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        t = range(num)
        if use_tqdm:
            t = tqdm(range(num), desc="Arena.playGames (1)")
        for _ in t:
            gameResult = self.playGame(verbose=verbose, print_final_board=print_final_board)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        t = range(num)
        if use_tqdm:
            t = tqdm(range(num), desc="Arena.playGames (2)")

        for _ in t:
            gameResult = self.playGame(verbose=verbose, print_final_board=print_final_board)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
