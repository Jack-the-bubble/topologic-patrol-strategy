#!/usr/bin/env python

import numpy as np

from leader_follower.Cell import Cell
from leader_follower.strategy.Strategy import Strategy
from leader_follower.strategy.MetricStrategy import MetricStrategy
from leader_follower.game_generation import generate_cells, \
    generate_adjacence, generate_p_strategy


def simple_optimize():
    idx_array = [0, 1, 2]
    p_payoff_array = [-0.5, 0, -0.6]
    i_payoff_array = [0.5, 0, 0.5]
    d_array = [2, 1, 2]
    dim_array = [10, 15, 12]
    X0 = 1
    Y0 = -1

    env = generate_cells(idx_array, p_payoff_array, i_payoff_array, d_array,
                         dim_array)
    adjacence_array = generate_adjacence([(0, 1), (1, 2)], 3)
    patroller_strat = generate_p_strategy(adjacence_array)

    strategy = MetricStrategy(X0, Y0, env, adjacence_array, patroller_strat)
    # strategy = Strategy(X0, Y0, env, adjacence_array, patroller_strat)

    res = strategy.generate_strategy()

    return res


def optimize3by2():
    env_count = 6
    idx_array = np.arange(env_count).tolist()
    p_payoff_array = np.zeros(env_count).tolist()
    i_payoff_array = np.zeros(env_count).tolist()
    d_array = np.ones(env_count).tolist()
    dim_array = np.ndarray([10, 10, 10, 10, 10, 10])

    p_payoff_array[0] = 0.5
    p_payoff_array[env_count - 1] = 0.5

    i_payoff_array[0] = 0.4
    i_payoff_array[env_count - 1] = 0.6

    d_array[0] = 4
    d_array[env_count - 1] = 4

    X0 = 1
    Y0 = -1

    env = generate_cells(idx_array, p_payoff_array, i_payoff_array, d_array,
                         dim_array)
    adjacence_array = generate_adjacence([(0, 1), (1, 2), (2, 5), (0, 3),
                                          (3, 4), (4, 5)],
                                         env_count)
    patroller_strat = generate_p_strategy(adjacence_array)

    strategy = MetricStrategy(X0, Y0, env, adjacence_array, patroller_strat)

    res = strategy.optimize()

    return res


def optimize4by3():
    env_count = 12
    idx_array = np.arange(env_count).tolist()
    p_payoff_array = np.zeros(env_count).tolist()
    i_payoff_array = np.zeros(env_count).tolist()
    d_array = np.ones(env_count).tolist()
    dim_array = np.ndarray([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

    p_payoff_array[3] = 0.5
    p_payoff_array[env_count - 1] = 0.5

    i_payoff_array[3] = 0.4
    i_payoff_array[env_count - 1] = 0.6

    d_array[3] = 5
    d_array[env_count - 1] = 5

    X0 = 1
    Y0 = -1

    env = generate_cells(idx_array, p_payoff_array, i_payoff_array, d_array,
                         dim_array)
    adjacence_array = generate_adjacence([(0, 1), (1, 2), (2, 3), (2, 6),
                                          (0, 4), (4, 8), (8, 9), (9, 10),
                                          (10, 11), (10, 6)],
                                         env_count)
    patroller_strat = generate_p_strategy(adjacence_array)

    strategy = MetricStrategy(X0, Y0, env, adjacence_array, patroller_strat)

    res = strategy.optimize()

    return res


if __name__ == '__main__':
    res = simple_optimize()

    # res = optimize4by3()
    print(res)
    print('\n')

    print(res.x.reshape((3, 3)))
