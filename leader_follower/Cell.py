class Cell:
    """! Class defining the environment cell.

    It stores information about the cell like its index, payoffs for robot and
        intruder and penetration time.
    """

    def __init__(self, idx=0, p=0, i=0, d=0, dim=0) -> None:
        """! Construct a cell with given parameters.

        @param idx index of a cell in the environment
        @param p patroler (robot) payoff
        @param i intruder payoff
        @param d number of game rounds for intruder to penetrate
        """
        ## cell index counted row-wise from top-left corner row-by-row
        self.index = idx

        ## payoff for robot if intruder penetrates
        self.p_payoff = p

        ## payoff for intruder if intruder penetrates
        self.i_payoff = i

        ## how many rounds it takes for intruder to penetrate
        self.d = d

        ## size of cell
        self.dim = dim
