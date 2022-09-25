from abc import ABC, abstractmethod
import numpy as np

from leader_follower.Cell import Cell


class BaseStrategy(ABC):
    def __init__(self, x0, y0, environment: list[Cell],
                 adj_matrix: np.ndarray, p_strat: np.ndarray) -> None:
        ## list of cells the environment consists of
        self.environment = environment

        ## ndarray defining adjacent cells from environment field
        self.adjacence_matrix = adj_matrix

        ## matrix(?) with probabilities of how likely a robot is
        ##   to go from cell i to cell j
        self.robot_strategy = p_strat

        ## enum(?) defining intruder strategy - should it wait, break in to
        ##  cell w or give up
        self.intruder_strategy = None

        ## payoff to the robot if intruder is captured or doesn't attack
        self.X0 = x0

        ## payoff to the intruder if it's captured
        self.Y0 = y0

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def gamma(self,  h: int, w: int, i: int, j: int, probability=0) -> float:
        pass
