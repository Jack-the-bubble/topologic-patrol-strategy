import numpy as np

from leader_follower.Cell import Cell


def generate_cells(idx: list[int], p_payoff: list[float],
                   i_payoff: list[float], d: list[int], dim: list[int]):
    """! Generate an array of Cell objects defining the environment.

    @param idx array of indexes of cells in environment
    @param p_payoff array of payoffs for the patroler (robot)
    @param i_payoff array of payoffs for the intruder
    @param d array of rounds to wait while penetrating for each cell
    @param dim array of cell dimensions for corresponding cell indexes

    @return list of Cell objects with mapped given parameters
    """
    if dim is None:
        dim = np.zeros(len(idx)).tolist()

    environment = []
    for ar_idx, tup in enumerate(zip(idx, p_payoff, i_payoff, d, dim)):
        c = Cell(*tup)
        environment.append(c)

    return environment


def generate_adjacence(adjacence_list: list((int, int)), count: int):
    """! Generate matrix specifying which cells are adjacent.

    Matrix size is (count, count), coordinates specified in adjacence_list
        are set to 1, also the cell is adjacent to itself
        The rest of values are 0.

    adj(x, y) = adj(y, x)

    @param adjacence_list list of tuples to get information from
    @param count count of cells in the environment

    @return ndarray with adjacence information
    """

    adj = np.eye(count, count, dtype=int)  # robot can stay in current cell
    for tup in adjacence_list:
        adj[tup[0]][tup[1]] = 1
        adj[tup[1]][tup[0]] = 1

    return adj


def generate_p_strategy(adj_matrix: np.ndarray):
    """! Generate arbitrary patroler strategy as an example.

    Make sure all rules from the whitepaper are applied.
    Row index is starting cell, column index is destination cell

    @param adj_matrix ndarray with adjacence information about environment

    @return ndarray of probabilities for movement between every cell
    """
    row_len = len(adj_matrix[0])
    strat = np.zeros((row_len ** 2, 1))
    for idx, row in enumerate(adj_matrix):
        neighbours_count = sum(row)
        probability = 1 / neighbours_count
        for row_idx, cell in enumerate(row):
            if cell == 1:
                strat[idx*row_len + row_idx] = probability

    return strat
