# for now this strategy is assuming the discovery of intruder is only possible, if both parties are in the same cell
from textwrap import indent
from math import inf
import numpy as np

from scipy.optimize import LinearConstraint, minimize, Bounds, \
    NonlinearConstraint, shgo

from .Cell import Cell


class Strategy:
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

        ## starting point for robot in phase 2
        self.s = Cell()

        ## cell to omit for robot in phase 2
        self.q = Cell()

    def optimize(self):
        constraints, intruder_constraints = self.get_eq_constraints()
        bound = Bounds(0, 1)
        # bound = self.getSHGOBounds()
        # res = minimize(self.target_fun, self.robot_strategy, method='nelder-mead',
        #                options={'xatol': 1e-8, 'disp': True},
        #                bounds=bound, constraints=constraints)
        res = minimize(self.target_fun, self.robot_strategy, bounds=bound,
                       constraints=constraints + intruder_constraints)
        # res = shgo(self.target_fun, bounds=bound,
        #            constraints=constraints)

        # if minimalization failed, go to second phase of the algorithm

        if not res.success:
            print("Phase 1 failed, started phase 2")
            self.optimize_second_phase(constraints)

        return res

    def optimize_second_phase(self, constraints: list):
        """! generate n*n problems and choose one with biggest robot payoff.

        @return scipy result object with highest payoff for robot.
        """
        optimize_results = []

        for i in self.environment:
            if sum(self.adjacence_matrix[i.index]) == 0:
                continue

            for j in self.environment:
                self.s = i
                self.q = j
                res = self.optimize_single_case(constraints)
                optimize_results.append(res)

        return self.get_best_strategy(optimize_results)

    def optimize_single_case(self, constraints: list):
        """! Generate constraints and optimize single case for second phase.

        Generates list of constraints for intruder [eq 8] and uses
            target function [eq 9] to find single candidate for robot strategy.

        @return result of optimization as scipy object from optimize toolkit
        """
        bound = Bounds(0, 1)
        print(f"optimizing for pair ({self.s}, {self.q}).")

        intruder_constraints = self.get_second_phase_constraints()
        res = minimize(self.second_phase_target_fun, self.robot_strategy,
                       bounds=bound,
                       constraints=constraints + intruder_constraints)
        print(f"Optimization result is {res.success}")

        return res

    def target_fun(self, alpha: np.ndarray) -> float:
        """! Calculate target function for patroller. [eq 2]

        For now it uses simplified version of basic example with only 3 cells.
        """
        self.robot_strategy = alpha

        target_value = 0

        for cell in self.environment:
            # if cell.p_payoff > 0:  # check if cell is a target
            target_value = target_value + \
                self.single_capture_payoff(cell) + \
                self.single_miss_payoff(cell)

        return -(target_value)

    def second_phase_target_fun(self, alpha: np.ndarray):
        """! Target function used to optimize in second phase

        Method expects fields self.s and self.q to be set in advance in order
            to calculate function value properly.

        Because the whitepaper uses maximization and optimization toolkit
            provides minimalization algorithms, the method returns the result
            multiplied by -1

        @param alpha - required by optimization toolkit that the function has
            a parametr to pass the vector of possible solutions

        @return value of target function based on given alpha vector
        """
        self.robot_strategy = alpha

        x_sum = self.get_prob_excluded_sum_0(self.s, self.q)

        return -1 * (self.q.p_payoff * x_sum + self.X0 * (1 - x_sum))

    def getSHGOBounds(self):

        bounds = []
        for _ in self.environment:
            for _ in self.environment:
                bounds.append((0, 1))

        return bounds

    def get_best_strategy(self, strategy_list: list):
        """! Get best strategy based on robot payoff value

        @param strategy_list list of optimization results from all
            optimizations in second phase of the algorithm

        @param result object from scipy optimization toolkit with highest
            robot payoff
        """

        payoff_list = []
        for res in strategy_list:
            # if res.success:
            payoff = self.second_phase_target_fun(res.x)
            payoff_list.append(payoff)

            # else:
            #     payoff_list.append(-inf)

        index = payoff_list.index(max(payoff_list))

        return strategy_list[index]

    def single_capture_payoff(self, cell: Cell) -> float:
        """! Calculate part of target function [eq 3]

        Calculate single payoff for robot for capturing intruder
            trying to get into given cell.

        @param cell cell to calculate payoff for

        @return expected payoff for capturing the intruder in given cell
        """

        payoff = 0
        cell_count = len(self.environment)
        for i in range(cell_count):
            for j in range(cell_count):

                # if either i or j is equal to w, robot will capture
                #   the intruder - but removing this line breaks the result...
                if cell.index == i or cell.index == j:
                    continue

                payoff = payoff + 1 - self.gamma(cell.d, cell.index, i, j)

        return self.X0 * payoff

    def single_miss_payoff(self, cell: Cell) -> float:
        """! Calculate second part of target function [eq 4]

        Calculate single payoff for robot not capturing the intruder
            trying to get into given cell.

        @param cell cell to calculate payoff for

        @return expected payoff for not capturing intruder in given cell
        """

        payoff = 0
        cell_count = len(self.environment)
        for i in range(cell_count):
            for j in range(cell_count):

                # if either i or j is equal to w, robot will capture \
                #   the intruder - but removing this line
                #   breaks the result...
                if cell.index == i or cell.index == j:
                    continue

                payoff = payoff + self.gamma(cell.d, cell.index, i, j)

        return cell.p_payoff * payoff

    def gamma(self, h: int, w: int, i: int, j: int, probability=0) -> float:
        """! Calculate gamma [eq 1]

        Get probability that the robot will not visit cell `w` when going
            from `i` to `j` given number of `h` moves it takes
            to break into cell `w`.

        @param h number of rounds to calculate gamma for
        @param w cell index that we want to calculate gamma for
        @param i cell robot is starting from
        @param j cell robot is going to

        @return probability that robot will not visit cell `w` while going from
            `i` to `j`.
        """

        if h == 1:
            return self.get_robot_strategy(i, j)
        elif h > 1:
            gamma = probability
            for x in self.environment:
                if x.index != w:
                    current_gamma = self.gamma(h-1, w, i, x.index, gamma)
                    current_alpha = self.get_robot_strategy(x.index, j)
                    gamma = gamma + current_gamma * current_alpha

            return gamma

    def get_robot_strategy(self, i: int, j: int) -> float:
        """! Return a probability for the robot to go from cell `i` to `j`."""

        if not self.adjacence_matrix[i][j]:
            return 0

        else:
            return self.robot_strategy[i * len(self.environment) + j]

    def get_prob_constraints(self):
        """! Create constraints for each cell that probability sum is 1 [eq 5]
        """
        cell_count = len(self.environment)
        constraints = []

        for a in range(cell_count):
            eval_str = "lambda x: sum(x[{}*{}:({}+1)*{}]) - 1".format(
                a, cell_count, a, cell_count)
            constraints.append({'type': 'eq', 'fun': eval(eval_str)})

        return constraints

    def get_zero_constraints(self):
        """! Crete list of constraints for alpha=0

        Return constraints for alpha(i, j) = 0 for connections in topology
            that don't exist, based on adjacence matrix

        @return list of dictionaries accepted by numpy optimize.minimize() fun
        """
        constraints = []
        cell_count = len(self.environment)

        for i in range(cell_count):
            for j in range(cell_count):
                if self.adjacence_matrix[i][j] == 0:
                    eval_str = "lambda x: x[{}]".format(i*cell_count+j)
                    constraints.append({'type': 'eq', 'fun': eval(eval_str)})

        return constraints

    def get_intruder_constraints_0(self):
        """! Create Constratints making sure no attack is better than stay-out

        [eq 6]
        """
        constraints = []

        # search for cells worth attacking
        for c in self.environment:
            if c.i_payoff > 0:
                for z in self.environment:
                    excluded_sum = self.get_prob_excluded_sum_0(z, c)
                    eval_str = "lambda x: ({}*(1 - {}) + {}*{})".format(
                        self.Y0, float(excluded_sum), c.i_payoff,
                        float(excluded_sum))

                    constraints.append({'type': 'ineq', 'fun': eval(eval_str)})

        return constraints

    # def get_intruder_constraints_1(self):
    #     """! Create Constratints making sure no attack is better than stay-out

    #     [eq 6]
    #     """
    #     constraints = []

    #     # search for cells worth attacking
    #     for c in self.environment:
    #         if c.i_payoff > 0:
    #             for z in self.environment:
    #                 excluded_sum = self.get_prob_excluded_sum_1(z, c)
    #                 eval_str = "lambda x: {}*{} + {}*{}".format(
    #                     self.Y0, float(excluded_sum), c.i_payoff,
    #                     float(excluded_sum))

    #                 constraints.append({'type': 'ineq', 'fun': eval(eval_str)})

    #     return constraints

    def get_prob_excluded_sum_0(self, z: Cell, w: Cell):
        """! Get sum of gammas from z to all cells with exception of w [eq 7]
        """
        sum_prob = 0
        for c in self.environment:
            if c.index != w.index:
                sum_prob += self.gamma(w.d, w.index, z.index, c.index)

        return sum_prob

    # def get_prob_excluded_sum_1(self, z: Cell, w: Cell):
    #     """! Get sum of gammas from z to all cells with exception of w [eq 7]
    #     """
    #     sum_prob = 0
    #     for c in self.environment:
    #         if c.index != w.index:
    #             sum_prob += (1 - self.gamma(w.d, w.index, z.index, c.index))

    #     return sum_prob

    def get_eq_constraints(self):

        prob_cons = self.get_prob_constraints()
        zero_cons = self.get_zero_constraints()
        intruder_cons = self.get_intruder_constraints_0()
        return prob_cons + zero_cons, intruder_cons
        # return []

    def get_second_phase_constraints(self):
        """! Get list of constraints needed for second phase of the algorithm

        [eq 8] Express that no intruder action is more valuable to the intruder
            than action(s, q), there can be n*n such constraints.

        @return list of dictionaries with constraints accepted by scipy
            opitmize toolkit
        """
        constraints = []

        sum_sq = self.get_prob_excluded_sum_0(self.s, self.q)
        sq_payoff = self.Y0 * (1 - sum_sq) + self.q.i_payoff * sum_sq

        for z in self.environment:
            for w in self.environment:
                sum_zw = self.get_prob_excluded_sum_0(z, w)
                temp_payoff = self.Y0 * (1 - sum_zw) + w.i_payoff * sum_zw

                eval_str = "lambda x: {} - {}".format(temp_payoff, sq_payoff)

                constraints.append({'type': 'ineq', 'fun': eval(eval_str)})

        return constraints
